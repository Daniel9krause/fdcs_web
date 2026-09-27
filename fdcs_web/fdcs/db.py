"""SQLite storage for scan history, store orders and contact messages."""
import json
import sqlite3
from datetime import datetime

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,             -- camera | upload | api
    image_file TEXT NOT NULL,
    annotated_file TEXT,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    crop TEXT, location TEXT, notes TEXT,
    result_json TEXT NOT NULL,
    eat_status TEXT,                  -- safe | caution | unsafe | unknown
    verified_label TEXT,              -- label confirmed/corrected by a person
    owner TEXT                        -- random id of the browser that made the scan
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    customer_name TEXT NOT NULL, phone TEXT NOT NULL, email TEXT,
    address TEXT NOT NULL, payment_method TEXT NOT NULL,
    items_json TEXT NOT NULL, total REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    token TEXT, payment_ref TEXT, momo_txn_id TEXT, momo_sender TEXT,
    paid_at TEXT, admin_note TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    product_id TEXT PRIMARY KEY,
    price REAL NOT NULL,              -- Ghana price in GH₵
    price_usd_intl REAL,              -- app plans: price abroad in US$
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL, email TEXT NOT NULL, message TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    with sqlite3.connect(app.config["DATABASE"]) as conn:
        conn.executescript(SCHEMA)
        # upgrade databases created by older versions of the app
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scans)")}
        for col in ("eat_status", "verified_label", "owner"):
            if col not in cols:
                conn.execute(f"ALTER TABLE scans ADD COLUMN {col} TEXT")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(orders)")}
        for col in ("token", "payment_ref", "momo_txn_id", "momo_sender", "paid_at", "admin_note",
                    "country", "display_total"):
            if col not in cols:
                conn.execute(f"ALTER TABLE orders ADD COLUMN {col} TEXT")
    app.teardown_appcontext(close_db)


# ---------------------------------------------------------------- scans
def add_scan(source, image_file, annotated_file, result: dict, crop="", location="", notes="", owner="") -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO scans (created_at, source, image_file, annotated_file, label, confidence,"
        " severity, is_demo, crop, location, notes, result_json, eat_status, owner)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now(), source, image_file, annotated_file, result["label"], result["confidence"],
         result["severity"], int(result["is_demo"]), crop, location, notes, json.dumps(result),
         result.get("edibility", {}).get("status", "unknown"), owner))
    db.commit()
    return cur.lastrowid


def get_scan(scan_id):
    row = get_db().execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return None
    scan = dict(row)
    scan["result"] = json.loads(scan["result_json"])
    return scan


def list_scans(label=None, limit=200, owner=None):
    where, args = [], []
    if label:
        where.append("label = ?"); args.append(label)
    if owner is not None:
        where.append("owner = ?"); args.append(owner)
    sql = "SELECT * FROM scans" + (" WHERE " + " AND ".join(where) if where else "")
    rows = [dict(r) for r in get_db().execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))]
    for r in rows:
        r["result"] = json.loads(r["result_json"])
    return rows


def scan_stats(owner=None):
    db = get_db()
    w, a = ("WHERE owner = ?", (owner,)) if owner is not None else ("", ())
    total = db.execute(f"SELECT COUNT(*) FROM scans {w}", a).fetchone()[0]
    by_label = db.execute(f"SELECT label, COUNT(*) AS n FROM scans {w} GROUP BY label ORDER BY n DESC", a).fetchall()
    return {"total": total, "by_label": [dict(r) for r in by_label]}


def set_verified_label(scan_id, label, eat_status):
    db = get_db()
    db.execute("UPDATE scans SET verified_label = ?, eat_status = ? WHERE id = ?", (label, eat_status, scan_id))
    db.commit()


# ---------------------------------------------------------------- gallery
LABEL_SQL = "COALESCE(verified_label, label)"


def gallery_items(crop=None, label=None, eat=None, verified_only=False, limit=500, owner=None):
    where, args = [], []
    if owner is not None:
        where.append("owner = ?"); args.append(owner)
    if crop:
        where.append("COALESCE(NULLIF(crop, ''), 'Unspecified') = ?"); args.append(crop)
    if label:
        where.append(f"{LABEL_SQL} = ?"); args.append(label)
    if eat:
        where.append("COALESCE(eat_status, 'unknown') = ?"); args.append(eat)
    if verified_only:
        where.append("verified_label IS NOT NULL")
    sql = f"SELECT *, {LABEL_SQL} AS final_label FROM scans"
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = [dict(r) for r in get_db().execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))]
    for r in rows:
        r["category"] = json.loads(r["result_json"]).get("category", "")
    return rows


def gallery_counts(owner=None):
    db = get_db()
    w, a = ("WHERE owner = ?", (owner,)) if owner is not None else ("", ())
    crops = db.execute("SELECT COALESCE(NULLIF(crop, ''), 'Unspecified') AS crop, COUNT(*) AS n,"
                       " SUM(verified_label IS NOT NULL) AS verified, MAX(id) AS cover_id"
                       f" FROM scans {w} GROUP BY 1 ORDER BY n DESC", a).fetchall()
    labels = db.execute(f"SELECT {LABEL_SQL} AS label, COUNT(*) AS n FROM scans {w} GROUP BY 1 ORDER BY n DESC",
                        a).fetchall()
    covers = {r["id"]: r["image_file"] for r in db.execute(
        f"SELECT id, image_file FROM scans WHERE id IN (SELECT MAX(id) FROM scans {w} GROUP BY"
        " COALESCE(NULLIF(crop, ''), 'Unspecified'))", a)}
    total = db.execute(f"SELECT COUNT(*), SUM(verified_label IS NOT NULL) FROM scans {w}", a).fetchone()
    return {"crops": [dict(r, cover=covers.get(r["cover_id"])) for r in crops],
            "labels": [dict(r) for r in labels], "total": total[0], "verified": total[1] or 0}


def delete_scan(scan_id):
    db = get_db()
    row = db.execute("SELECT image_file, annotated_file FROM scans WHERE id = ?", (scan_id,)).fetchone()
    db.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    db.commit()
    return dict(row) if row else None


# ---------------------------------------------------------------- store
ORDER_STATUSES = ("awaiting_payment", "payment_submitted", "paid", "cash_on_delivery", "delivered",
                  "cancelled", "confirmed")


def add_order(customer: dict, items: list, total: float, status: str, token: str,
              country: str = "GH", display_total: str = "") -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO orders (created_at, customer_name, phone, email, address, payment_method,"
        " items_json, total, status, token, country, display_total) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now(), customer["name"], customer["phone"], customer.get("email", ""),
         customer["address"], customer["payment_method"], json.dumps(items), total, status, token,
         country, display_total))
    db.commit()
    return cur.lastrowid


def update_order(order_id, **fields):
    allowed = {"status", "payment_ref", "momo_txn_id", "momo_sender", "paid_at", "admin_note"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    db = get_db()
    db.execute(f"UPDATE orders SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
               (*fields.values(), order_id))
    db.commit()


def find_order_by_ref(reference):
    row = get_db().execute("SELECT id FROM orders WHERE payment_ref = ?", (reference,)).fetchone()
    return get_order(row["id"]) if row else None


def list_orders(status=None):
    sql, args = "SELECT * FROM orders", []
    if status:
        sql, args = sql + " WHERE status = ?", [status]
    rows = [dict(r) for r in get_db().execute(sql + " ORDER BY id DESC", args)]
    for r in rows:
        r["items"] = json.loads(r["items_json"])
    return rows


def order_totals():
    db = get_db()
    paid = db.execute("SELECT COALESCE(SUM(total), 0), COUNT(*) FROM orders WHERE status IN"
                      " ('paid', 'delivered')").fetchone()
    waiting = db.execute("SELECT COUNT(*) FROM orders WHERE status IN ('awaiting_payment',"
                         " 'payment_submitted')").fetchone()[0]
    return {"paid_total": paid[0], "paid_count": paid[1], "to_check": waiting}


def list_messages():
    return [dict(r) for r in get_db().execute("SELECT * FROM messages ORDER BY id DESC LIMIT 200")]


def get_order(order_id):
    row = get_db().execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        return None
    order = dict(row)
    order["items"] = json.loads(order["items_json"])
    return order


def add_message(name, email, message):
    db = get_db()
    db.execute("INSERT INTO messages (created_at, name, email, message) VALUES (?,?,?,?)",
               (now(), name, email, message))
    db.commit()


# ---------------------------------------------------------------- prices
def price_overrides(conn=None):
    c = conn or get_db()
    return {r[0]: (r[1], r[2]) for r in c.execute("SELECT product_id, price, price_usd_intl FROM prices")}


def set_price(product_id, price, price_usd_intl=None):
    db = get_db()
    db.execute("INSERT INTO prices (product_id, price, price_usd_intl, updated_at) VALUES (?,?,?,?)"
               " ON CONFLICT(product_id) DO UPDATE SET price = excluded.price,"
               " price_usd_intl = excluded.price_usd_intl, updated_at = excluded.updated_at",
               (product_id, price, price_usd_intl, now()))
    db.commit()
