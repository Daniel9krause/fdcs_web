"""Run with:  python -m pytest -q"""
import base64
import io
import os
import sys

import numpy as np
import json
import re
import sqlite3
import zipfile

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app  # noqa: E402
from fdcs.detector import analyze, annotate  # noqa: E402
from fdcs.safety import assess  # noqa: E402


def jpeg_bytes(color=(60, 140, 60), size=(640, 480)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def make_app(tmp_path, name="t.db", secure=False, **kw):
    """Functional tests switch CSRF/rate limits off; security tests switch them on."""
    opts = dict(DATABASE=str(tmp_path / name), UPLOAD_DIR=str(tmp_path), MODEL_PATH=str(tmp_path / "missing.tflite"),
                TESTING=True, ADMIN_PASSWORD="owner-pass", PAYSTACK_SECRET_KEY="",
                CSRF_DISABLED=not secure, RATE_LIMITS_DISABLED=not secure)
    opts.update(kw)
    return create_app(**opts)


def admin_login(c):
    c.post("/admin/login", data={"password": "owner-pass"})


@pytest.fixture()
def client(tmp_path):
    return make_app(tmp_path).test_client()


@pytest.mark.parametrize("url", ["/", "/scan", "/gallery", "/pests?category=disease", "/pests/aspergillus_mould", "/pests", "/pests/fall_armyworm", "/store",
                                 "/store?category=Tools", "/store/neem-oil", "/cart", "/history",
                                 "/about", "/contact", "/api/health", "/api/pests"])
def test_pages_load(client, url):
    assert client.get(url).status_code == 200


def test_unknown_pages_404(client):
    assert client.get("/pests/unicorn").status_code == 404
    assert client.get("/store/nothing").status_code == 404
    assert client.get("/result/999").status_code == 404


def test_upload_scan_flow(client):
    r = client.post("/scan", data={"image": (io.BytesIO(jpeg_bytes()), "leaf.jpg"), "crop": "Maize"},
                    content_type="multipart/form-data")
    assert r.status_code == 302 and "/result/" in r.location
    page = client.get(r.location)
    assert page.status_code == 200 and b"Top predictions" in page.data
    assert b"Maize" in client.get("/history").data


def test_camera_scan_flow(client):
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes()).decode()
    r = client.post("/scan", data={"camera_image": data_url})
    assert r.status_code == 302


def test_rejects_bad_input(client):
    assert b"Please take a photo" in client.post("/scan", data={}).data
    r = client.post("/scan", data={"image": (io.BytesIO(b"nope"), "x.jpg")}, content_type="multipart/form-data")
    assert b"not a valid image" in r.data
    r = client.post("/scan", data={"image": (io.BytesIO(b"nope"), "x.exe")}, content_type="multipart/form-data")
    assert b"Unsupported file type" in r.data


def test_api_predict(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(jpeg_bytes()), "a.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["result"]["is_demo"] is True
    assert body["safe_to_eat"]["status"] in ("safe", "caution", "unsafe", "unknown")
    assert body["organism"]["common_name"] and "kingdom" in body["organism"]
    assert body["category"] in ("pest", "disease", "healthy", "unrecognised")
    assert client.post("/api/predict").status_code == 400


def test_cart_and_checkout(client):
    client.post("/cart/add/neem-oil", data={"qty": 2})
    client.post("/cart/add/plan-pro")
    assert b"430.00" in client.get("/cart").data
    bad = client.post("/checkout", data={"name": "K", "phone": "1", "address": "", "payment_method": "x"})
    assert bad.data.count(b"flash-error") >= 3
    ok = client.post("/checkout", data={"name": "Ama Owusu", "phone": "024 123 4567",
                                        "address": "GA-123-4567", "payment_method": "momo"})
    assert ok.status_code == 302
    page = client.get(ok.location).data
    assert b"Thank you, Ama" in page and b"0200687324" in page and b"430.00" in page
    assert b"Your cart is empty" in client.get("/cart").data


class FakeClassifier:
    """Says 'fall_armyworm' for any crop containing the brown patch, else 'healthy'."""
    labels = ["fall_armyworm", "healthy"]
    is_demo = False

    def predict_proba(self, img):
        arr = np.asarray(img, dtype=np.float32)
        brown = ((arr[..., 0] > 100) & (arr[..., 1] < 90)).mean()
        return np.array([0.9, 0.1]) if brown > 0.02 else np.array([0.05, 0.95])


def test_detector_localises_pest():
    img = Image.new("RGB", (900, 900), (50, 150, 50))
    img.paste((140, 70, 30), (650, 650, 850, 850))  # pest in bottom-right corner
    result = analyze(img, FakeClassifier(), grid=3)
    assert result.label == "fall_armyworm"
    assert result.detections, "expected at least one box"
    x1, y1, x2, y2 = result.detections[0].box
    # box covers the patch and excludes the healthy top-left region
    assert x1 <= 650 and y1 <= 650 and x2 >= 850 and y2 >= 850
    assert x1 >= 400 and y1 >= 400
    assert annotate(img, result, {"fall_armyworm": "Fall Armyworm"}).size == img.size


def test_detector_healthy():
    result = analyze(Image.new("RGB", (600, 600), (50, 150, 50)), FakeClassifier())
    assert result.label == "healthy" and result.severity == "none" and not result.detections


def test_detector_separates_two_regions():
    img = Image.new("RGB", (900, 600), (50, 150, 50))
    img.paste((140, 70, 30), (40, 40, 140, 120))     # top-left pest
    img.paste((140, 70, 30), (760, 470, 860, 560))   # bottom-right pest
    result = analyze(img, FakeClassifier(), grid=3)
    assert len(result.detections) == 2


# ------------------------------------------------------------ knowledge base
def test_every_label_has_complete_entry():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pests = json.load(open(os.path.join(root, "data/pests.json"), encoding="utf-8"))
    labels = open(os.path.join(root, "models/labels.txt")).read().split()
    products = {p["id"] for p in json.load(open(os.path.join(root, "data/products.json"), encoding="utf-8"))}
    assert set(labels) <= set(pests), set(labels) - set(pests)  # every model class has a profile
    for k, v in pests.items():
        assert v["category"] in ("pest", "disease", "healthy"), k
        assert v["edibility"]["status"] in ("safe", "caution", "unsafe"), k
        assert v["edibility"]["advice"], k
        assert set(v["products"]) <= products, k


# ------------------------------------------------------------ food safety
def test_safety_verdicts():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pests = json.load(open(os.path.join(root, "data/pests.json"), encoding="utf-8"))
    assert assess(pests["aspergillus_mould"], False)["status"] == "unsafe"
    assert assess(pests["healthy"], False)["status"] == "safe"
    sprayed = assess(pests["healthy"], False, sprayed="yes")
    assert sprayed["status"] == "caution" and "pre-harvest" in sprayed["advice"][0]
    assert assess(pests["aspergillus_mould"], False, sprayed="yes")["status"] == "unsafe"  # never upgraded
    assert assess(pests["healthy"], True)["status"] == "unknown"  # low confidence


# ------------------------------------------------------------ gallery
def test_gallery_bulk_upload_verify_and_export(client):
    files = [(io.BytesIO(jpeg_bytes((50 + i * 30, 120, 60))), f"t{i}.jpg") for i in range(3)]
    files.append((io.BytesIO(b"junk"), "bad.jpg"))
    r = client.post("/gallery/upload", data={"crop": "Tomato", "images": files},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"Saved and analysed 3 images" in r.data and b"skipped 1" in r.data
    page = client.get("/gallery?crop=Tomato")
    assert page.data.count(b'class="photo"') == 3

    assert client.get("/gallery/export.zip").status_code == 302          # export is owner-only
    admin_login(client)
    assert client.get("/gallery/export.zip?verified=1").status_code == 302  # nothing verified yet
    client.post("/result/1/verify", data={"verified_label": "tomato_late_blight"})
    r = client.get("/gallery/export.zip?verified=1")
    assert r.status_code == 200 and r.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    names = zf.namelist()
    assert "metadata.csv" in names
    assert [n for n in names if n.startswith("tomato_late_blight/")] and len(names) == 2
    # "all images" export: verified photos + confident predictions, never unconfirmed low-confidence guesses
    with client.application.app_context():
        from fdcs import db as fdb
        rows = fdb.gallery_items()
        expected = [r for r in rows if r["verified_label"] or r["category"] != "unrecognised"]
    everything = zipfile.ZipFile(io.BytesIO(client.get("/gallery/export.zip").data)).namelist()
    assert len(everything) == len(expected) + 1

    # confirming a dangerous condition makes the stored verdict "unsafe" and the result page follows it
    client.post("/result/2/verify", data={"verified_label": "aspergillus_mould"})
    assert b"eat-unsafe" in client.get("/gallery?crop=Tomato").data
    page = client.get("/result/2").data
    assert b"Not safe to eat" in page and b"Confirmed by a person" in page
    assert client.post("/result/1/verify", data={"verified_label": "hacker"}).status_code == 400


def test_scan_stores_safety_and_crop(client):
    r = client.post("/scan", data={"image": (io.BytesIO(jpeg_bytes()), "leaf.jpg"), "crop": "Cassava",
                                   "sprayed": "yes"}, content_type="multipart/form-data")
    page = client.get(r.location)
    assert b"Is the produce safe to eat?" in page.data
    assert b"Cassava" in client.get("/gallery").data


def test_old_database_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY, created_at TEXT, source TEXT, image_file TEXT,"
                     " annotated_file TEXT, label TEXT, confidence REAL, severity TEXT, is_demo INTEGER,"
                     " crop TEXT, location TEXT, notes TEXT, result_json TEXT)")
    make_app(tmp_path, DATABASE=str(path))
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scans)")}
    assert {"eat_status", "verified_label"} <= cols


# ------------------------------------------------------------ payments
def place_order(client, method="momo", email="", add=True):
    if add:
        client.post("/cart/add/neem-oil", data={"qty": 1})
    r = client.post("/checkout", data={"name": "Kofi Mensah", "phone": "0241234567", "email": email,
                                       "address": "GA-123-4567", "payment_method": method})
    return r


def test_order_page_needs_secret_token(client):
    loc = place_order(client).location
    order_id = loc.split("/")[2]
    assert client.get(f"/order/{order_id}/wrong-token").status_code == 404
    assert client.get(loc).status_code == 200


def test_direct_momo_flow_and_admin(client):
    loc = place_order(client).location
    assert b"Send GH" in client.get(loc).data
    bad = client.post(loc, data={"momo_txn_id": "", "momo_sender": "x"})
    assert b"flash-error" in bad.data
    r = client.post(loc, data={"momo_txn_id": "TX123456789", "momo_sender": "024 123 4567"}, follow_redirects=True)
    assert b"checking the payment" in r.data

    assert client.get("/admin").status_code == 302                      # must log in
    assert b"Wrong password" in client.post("/admin/login", data={"password": "nope"}).data
    client.post("/admin/login", data={"password": "owner-pass"})
    admin = client.get("/admin").data
    assert b"TX123456789" in admin and b"Check payment" in admin
    client.post("/admin/orders/1", data={"action": "paid"})
    assert b"Payment received" in client.get(loc).data
    assert b"90.00" in client.get("/admin").data                        # counted as received


def test_admin_disabled_without_password(tmp_path):
    app = make_app(tmp_path, "a.db", ADMIN_PASSWORD="")
    c = app.test_client()
    assert b"switched off" in c.post("/admin/login", data={"password": ""}).data
    assert c.get("/admin").status_code == 302


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_paystack_flow(tmp_path):
    from fdcs.payments import Paystack
    calls = []

    def opener(req, timeout=20):
        calls.append((req.get_method(), req.full_url, req.data))
        if req.full_url.endswith("/transaction/initialize"):
            body = json.loads(req.data)
            assert body["amount"] == 9000 and body["currency"] == "GHS"
            return FakeResponse(json.dumps({"status": True, "data": {"authorization_url": "https://checkout.paystack.com/abc"}}).encode())
        return FakeResponse(json.dumps({"status": True, "data": {"status": "success", "currency": "GHS",
                                        "amount": 9000, "paid_at": "2026-09-26T21:00:00Z", "channel": "mobile_money"}}).encode())

    app = make_app(tmp_path, "p.db", PAYSTACK_SECRET_KEY="sk_test_x")
    app.extensions["fdcs_paystack"] = Paystack("sk_test_x", opener=opener)
    c = app.test_client()
    c.post("/cart/add/neem-oil")
    assert b"Pay now with Mobile Money or card" in c.get("/checkout").data
    r = place_order(c, "paystack", email="kofi@example.com", add=False)
    assert r.location == "https://checkout.paystack.com/abc"
    with app.app_context():
        from fdcs import db as fdb
        ref = fdb.get_order(1)["payment_ref"]
    r = c.get(f"/payment/paystack/callback?reference={ref}", follow_redirects=True)
    assert b"Payment received" in r.data

    # webhook: rejects bad signatures, accepts Paystack-signed ones
    import hashlib, hmac
    body = json.dumps({"event": "charge.success", "data": {"reference": ref}}).encode()
    assert c.post("/payment/paystack/webhook", data=body, headers={"x-paystack-signature": "bad"}).status_code == 401
    sig = hmac.new(b"sk_test_x", body, hashlib.sha512).hexdigest()
    assert c.post("/payment/paystack/webhook", data=body, headers={"x-paystack-signature": sig},
                  content_type="application/json").status_code == 200


def test_paystack_underpayment_not_accepted():
    from fdcs.payments import Paystack
    def opener(req, timeout=20):
        return FakeResponse(json.dumps({"status": True, "data": {"status": "success", "currency": "GHS",
                                        "amount": 100, "paid_at": None}}).encode())
    assert Paystack("k", opener=opener).verify("R", 85.0)["paid"] is False


# ------------------------------------------------------------ security
def csrf_from(page: bytes) -> str:
    m = re.search(rb'name="_csrf" value="([^"]+)"', page)
    assert m, "form has no CSRF token"
    return m.group(1).decode()


def test_csrf_blocks_forged_forms(tmp_path):
    c = make_app(tmp_path, secure=True).test_client()
    assert c.post("/contact", data={"name": "A", "email": "a@b.co", "message": "hello"}).status_code == 400
    token = csrf_from(c.get("/contact").data)
    r = c.post("/contact", data={"name": "A", "email": "a@b.co", "message": "hello", "_csrf": token})
    assert r.status_code == 302
    other = make_app(tmp_path, "o.db", secure=True).test_client()   # token from another session is useless
    assert other.post("/contact", data={"name": "A", "email": "a@b.co", "message": "hi!!", "_csrf": token}).status_code == 400


def test_security_headers(client):
    r = client.get("/")
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert r.headers["X-Frame-Options"] == "DENY" and r.headers["X-Content-Type-Options"] == "nosniff"
    cookie = r.headers.get("Set-Cookie", "")
    assert "fdcs_session=" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie
    for page in ("/", "/scan", "/store", "/gallery", "/pests"):
        html = client.get(page).data
        assert b"<script>" not in html and b"onclick=" not in html and b"onsubmit=" not in html


def test_admin_login_rate_limited(tmp_path):
    c = make_app(tmp_path, secure=True).test_client()
    codes = []
    for _ in range(7):
        token = csrf_from(c.get("/admin/login").data)
        codes.append(c.post("/admin/login", data={"password": "guess", "_csrf": token}).status_code)
    assert 429 in codes and codes[:5] == [200] * 5


def test_visitors_cannot_see_each_others_scans(tmp_path):
    app = make_app(tmp_path)
    alice, bob = app.test_client(), app.test_client()
    r = alice.post("/scan", data={"image": (io.BytesIO(jpeg_bytes()), "a.jpg"), "location": "Alice farm"},
                   content_type="multipart/form-data")
    url = r.location
    assert alice.get(url).status_code == 200
    assert bob.get(url).status_code == 404                                   # can't open
    assert bob.post(url + "/verify", data={"verified_label": "healthy"}).status_code == 404
    assert bob.post("/history/1/delete").status_code == 404                  # can't delete
    assert b"Alice farm" not in bob.get("/history").data and b"Alice farm" in alice.get("/history").data
    assert b'class="photo"' not in bob.get("/gallery").data
    admin_login(bob)
    assert bob.get(url).status_code == 200 and b'class="photo"' in bob.get("/gallery").data


def test_open_redirect_blocked(client):
    for evil in ("https://evil.com", "//evil.com", "/\\evil.com"):
        r = client.post("/cart/add/neem-oil", data={"next": evil})
        assert r.location == "/cart", evil


def test_uploads_route_rejects_other_files(client):
    for bad in ("../app.py", "..%2Fapp.py", "fdcs.sqlite3", "secret_key", "x.jpg"):
        assert client.get(f"/uploads/{bad}").status_code == 404


def test_decompression_bomb_rejected(client):
    buf = io.BytesIO()
    Image.new("1", (9000, 9000)).save(buf, "PNG")  # tiny file, 81 million pixels
    r = client.post("/scan", data={"image": (io.BytesIO(buf.getvalue()), "bomb.png")},
                    content_type="multipart/form-data")
    assert b"not a valid image" in r.data


def test_csv_export_neutralises_formulas():
    from app import csv_safe
    assert csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)" and csv_safe("Ejura") == "Ejura"


def test_api_key_when_configured(tmp_path):
    c = make_app(tmp_path, API_KEY="k123").test_client()
    data = lambda: {"image": (io.BytesIO(jpeg_bytes()), "a.jpg")}
    assert c.post("/api/predict", data=data(), content_type="multipart/form-data").status_code == 401
    assert c.post("/api/predict", data=data(), content_type="multipart/form-data",
                  headers={"X-API-Key": "k123"}).status_code == 200


def test_secret_key_is_random_and_persistent(tmp_path):
    from fdcs.security import load_secret_key
    k1 = load_secret_key("change-me-in-production", str(tmp_path))
    assert len(k1) == 64 and k1 == load_secret_key("", str(tmp_path))


# ------------------------------------------------------------ pricing
def test_paid_plans_above_200_cedis_and_35_dollars_abroad():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plans = [p for p in json.load(open(os.path.join(root, "data/products.json"), encoding="utf-8"))
             if p["category"] == "App Plans" and p["price"] > 0]
    assert plans and all(p["price"] > 200 for p in plans)
    assert next(p for p in plans if p["id"] == "plan-pro")["price_usd_intl"] == 35


def test_visitor_abroad_sees_local_currency_and_pays_cedis(tmp_path):
    c = make_app(tmp_path).test_client()
    ng = {"Accept-Language": "en-NG,en;q=0.9"}
    page = c.get("/store/plan-pro", headers=ng).data.decode()
    assert "₦46,514" in page and "US$35" in page                 # 35 x 1328.96 NGN per US$
    assert "₦" not in c.get("/store/plan-pro").data.decode()     # Ghana visitor sees cedis
    r = c.post("/cart/add/knapsack-sprayer", headers=ng, follow_redirects=True)
    assert "delivered in Ghana only" in r.data.decode()
    c.post("/cart/add/plan-pro", headers=ng)
    checkout = c.get("/checkout", headers=ng).data.decode()
    assert "₦46,514" in checkout and "GH₵404.87" in checkout and 'value="cod"' not in checkout
    r = c.post("/checkout", headers=ng, data={"name": "Tunde Bello", "phone": "+234 803 123 4567",
                                              "address": "Ikeja, Lagos, Nigeria", "payment_method": "momo"})
    with c.application.app_context():
        from fdcs import db as fdb
        order = fdb.get_order(1)
    assert order["total"] == 404.87 and order["country"] == "NG" and order["display_total"] == "₦46,514"


def test_switching_country_removes_ghana_only_items(client):
    client.post("/cart/add/neem-oil")
    client.post("/cart/add/plan-pro")
    r = client.post("/country", data={"country": "GB", "next": "/cart"}, follow_redirects=True).data.decode()
    assert "Removed items" in r and "Neem" not in r.split("Your cart")[1].split("Not included")[0]
    assert "£" in r
    assert client.post("/country", data={"country": "ZZ"}).status_code == 302   # unknown code ignored


def test_owner_can_change_prices(tmp_path):
    app = make_app(tmp_path)
    c = app.test_client()
    assert c.get("/admin/prices").status_code == 302                             # owner only
    admin_login(c)
    page = c.get("/admin/prices").data.decode()
    assert "US$1 = GH₵" in page
    form = {f"price_{p['id']}": p["price"] for p in json.load(open(os.path.join(app.root_path, "data/products.json")))}
    form.update({"price_plan-pro": "300", "usd_plan-pro": "40", "usd_plan-free": "0", "usd_plan-coop": "150"})
    assert "Saved 1 price change" in c.post("/admin/prices", data=form, follow_redirects=True).data.decode()
    assert "GH₵ 300.00" in c.get("/store/plan-pro").data.decode()
    again = make_app(tmp_path).test_client()                                     # survives restart
    assert "GH₵ 300.00" in again.get("/store/plan-pro").data.decode()
    form["price_plan-pro"] = "-5"
    assert "Check the prices" in c.post("/admin/prices", data=form, follow_redirects=True).data.decode()


def test_exchange_rates_refresh_and_fallback(tmp_path):
    from fdcs.pricing import Rates
    good = lambda: {"result": "success", "time_last_update_utc": "today", "rates": {"GHS": 12.0, "NGN": 1500}}
    r = Rates(str(tmp_path / "rates.json"), fetch=good, auto_refresh=False)
    assert r.refresh(force=True) and r.per_usd("GHS") == 12.0 and r.source == "live"
    assert Rates(str(tmp_path / "rates.json"), auto_refresh=False).per_usd("NGN") == 1500   # cached to disk

    def broken():
        raise OSError("offline")
    r2 = Rates(str(tmp_path / "none.json"), fetch=broken, auto_refresh=False)
    assert r2.refresh(force=True) is False and r2.per_usd("GHS") > 10 and r2.source == "built-in"
    bad = lambda: {"result": "success", "rates": {"GHS": 0}}                                # nonsense rejected
    r3 = Rates(str(tmp_path / "bad.json"), fetch=bad, auto_refresh=False)
    assert r3.refresh(force=True) is False


# ------------------------------------------------------------ installable web app (PWA)
def test_manifest_is_installable(client):
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.mimetype == "application/manifest+json"
    m = r.get_json()
    assert m["display"] == "standalone" and m["start_url"].startswith("/") and m["name"]
    sizes = {i["sizes"] for i in m["icons"]}
    assert {"192x192", "512x512"} <= sizes and any(i.get("purpose") == "maskable" for i in m["icons"])
    for icon in m["icons"]:
        assert client.get(icon["src"]).status_code == 200


def test_service_worker_served_from_root(client):
    r = client.get("/sw.js")
    assert r.status_code == 200 and "javascript" in r.mimetype
    assert r.headers["Service-Worker-Allowed"] == "/" and "no-cache" in r.headers["Cache-Control"]
    body = r.data.decode()
    assert "/offline" in body and "admin" in body          # offline fallback, private pages excluded
    for path in ("/offline", "/static/css/style.css", "/static/js/app.js", "/static/icons/icon-192.png"):
        assert client.get(path).status_code == 200, path    # everything the worker pre-caches exists


def test_pages_link_manifest_and_icons(client):
    html = client.get("/").data.decode()
    for needle in ('rel="manifest"', 'name="theme-color"', 'rel="apple-touch-icon"', "viewport-fit=cover",
                   "data-install"):
        assert needle in html, needle


def test_owner_pages_never_cached(client):
    assert "no-store" not in client.get("/store").headers.get("Cache-Control", "")
    admin_login(client)
    assert "no-store" in client.get("/store").headers["Cache-Control"]   # owner session: nothing stored


def test_launcher_finds_tunnel_link():
    import launcher
    line = "INF |  https://crops-tomato-demo.trycloudflare.com       |"
    assert launcher.URL_RE.search(line).group(0) == "https://crops-tomato-demo.trycloudflare.com"


def test_serve_banner_address():
    import serve
    ip = serve.lan_address()
    assert ip is None or (ip.count(".") == 3 and not ip.startswith("127."))
