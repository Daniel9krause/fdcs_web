"""FDCS web application - pest detection, pest library and online store.

Run:  python app.py        then open http://127.0.0.1:5000
"""
import base64
import binascii
import csv
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import tempfile
import time
import uuid
import warnings
import zipfile
from functools import wraps

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, session, url_for)
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from fdcs import db, load_json
from fdcs.detector import analyze, annotate, load_image
from fdcs.model import PestClassifier
from fdcs.payments import PaymentError, Paystack, new_reference
from fdcs.pricing import COUNTRIES, HOME, Rates, country_from_accept_language, fmt, quote
from fdcs.safety import VERDICTS, assess
from fdcs.security import init_security, rate_limit, safe_next, visitor_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
def csv_safe(value):
    """Stop spreadsheet formula injection when the exported CSV is opened in Excel."""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


PHONE_RE = re.compile(r"^\+?[0-9 ]{9,18}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app(config_object=Config, **overrides):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.config.update(overrides)
    init_security(app)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    db.init_db(app)

    pests = load_json(app.config["PESTS_FILE"])
    products = load_json(app.config["PRODUCTS_FILE"])
    product_map = {p["id"]: p for p in products}
    base_prices = {p["id"]: (p["price"], p.get("price_usd_intl")) for p in products}
    rates = app.extensions.get("fdcs_rates") or Rates(
        os.path.join(app.root_path, "instance", "rates.json"),
        auto_refresh=not app.config.get("TESTING"))
    app.extensions["fdcs_rates"] = rates

    def apply_price_overrides():
        """Owner-edited prices (admin > Prices) replace the defaults from products.json."""
        with sqlite3.connect(app.config["DATABASE"]) as conn:
            overrides = db.price_overrides(conn)
        for pid, p in product_map.items():
            price, usd = overrides.get(pid, base_prices[pid])
            p["price"] = price
            if p["category"] == "App Plans":
                p["price_usd_intl"] = usd if usd is not None else base_prices[pid][1]

    apply_price_overrides()

    def current_country():
        code = session.get("country")
        if code not in COUNTRIES:
            code = country_from_accept_language(request.headers.get("Accept-Language", ""))
        return code

    def price(product):
        return quote(product, current_country(), rates)
    classifier = PestClassifier(app.config["MODEL_PATH"], app.config["LABELS_PATH"],
                                app.config["INPUT_SIZE"], app.config["NORMALIZE"])
    app.extensions["fdcs_classifier"] = classifier
    names = {k: v["common_name"] for k, v in pests.items()}

    # ------------------------------------------------------------- helpers
    def pest_info(label):
        return pests.get(label, {"common_name": label.replace("_", " ").title(),
                                 "scientific_name": "-", "organism_type": "Unknown",
                                 "severity": "unknown", "description": "No library entry yet.",
                                 "crops": [], "identification": [], "symptoms": [],
                                 "lifecycle": "-", "order": "-", "category": "unknown",
                                 "kingdom": "-", "group": "Unknown", "taxonomy": "-",
                                 "management": {"cultural": [], "biological": [], "chemical": []},
                                 "products": []})

    def is_admin():
        """Admin sessions last 8 hours, and only work while an admin password is configured."""
        fresh = time.time() - session.get("admin_at", 0) < 8 * 3600
        return bool(session.get("is_admin")) and fresh and bool(app.config.get("ADMIN_PASSWORD"))

    def admin_required(view):
        @wraps(view)
        def wrapper(*a, **kw):
            if not is_admin():
                return redirect(url_for("admin_login", next=request.path))
            return view(*a, **kw)
        return wrapper

    def scope():
        """Owner filter: admins see every scan, visitors only their own."""
        return None if is_admin() else visitor_id()

    def own_scan_or_404(scan_id):
        record = db.get_scan(scan_id)
        if record is None or not (is_admin() or (record.get("owner") and record["owner"] == visitor_id())):
            abort(404)
        return record

    def allowed(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

    def read_submitted_image():
        """Return (bytes, source) from a file upload or a camera data-URL, or raise ValueError."""
        file = request.files.get("image")
        if file and file.filename:
            if not allowed(file.filename):
                raise ValueError("Unsupported file type. Use JPG, PNG, WEBP or BMP.")
            return file.read(), "upload"
        data_url = request.form.get("camera_image", "")
        if data_url.startswith("data:image/"):
            try:
                return base64.b64decode(data_url.split(",", 1)[1]), "camera"
            except (IndexError, binascii.Error):
                raise ValueError("Camera image could not be read.")
        raise ValueError("Please take a photo or choose an image first.")

    def run_detection(raw: bytes, source: str, crop="", location="", notes="", sprayed="no", owner=None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                img = load_image(raw)
        except Exception:
            raise ValueError("That file is not a valid image.")
        result = analyze(img, classifier, app.config["HEALTHY_LABEL"], app.config["TILE_GRID"],
                         app.config["TILE_THRESHOLD"], app.config["MIN_CONFIDENCE"])
        stem = uuid.uuid4().hex
        original, marked = f"{stem}.jpg", f"{stem}_det.jpg"
        img.save(os.path.join(app.config["UPLOAD_DIR"], original), quality=90)
        annotate(img, result, names).save(os.path.join(app.config["UPLOAD_DIR"], marked), quality=90)
        data = result.to_dict()
        info = pest_info(result.label)
        data["category"] = "unrecognised" if result.uncertain else info.get("category", "unknown")
        data["organism"] = {k: info.get(k, "-") for k in
                            ("common_name", "scientific_name", "organism_type", "kingdom", "group", "taxonomy", "order")}
        data["edibility"] = assess(info, result.uncertain, sprayed if sprayed in ("yes", "unsure") else "no",
                                   classifier.is_demo)
        scan_id = db.add_scan(source, original, marked, data, crop, location, notes,
                              owner if owner is not None else visitor_id())
        return scan_id, data

    def cart():
        return session.setdefault("cart", {})

    def cart_lines():
        """Cart lines priced for the visitor's country. `total` is what is charged, in GH₵."""
        lines, total, local = [], 0.0, 0.0
        country = current_country()
        _, cur, sym, dec = COUNTRIES[country]
        for pid, qty in cart().items():
            product = product_map.get(pid)
            if not product:
                continue
            q = price(product)
            if not q["available"]:
                continue
            subtotal = round(q["charge_ghs"] * qty, 2)
            total += subtotal
            local += q["value"] * qty
            lines.append({"product": product, "qty": qty, "subtotal": subtotal, "unit": q["display"],
                          "subtotal_display": ("Free" if not subtotal else
                                               fmt(subtotal, "GH₵ ", 2) if country == HOME else fmt(q["value"] * qty, sym, dec))})
        total = round(total, 2)
        if country == HOME:
            display_total = fmt(total, "GH₵ ", 2) if total else "Free"
        else:
            display_total = fmt(local, sym, dec) if total else "Free"
        return lines, total, display_total

    def cart_blocked():
        """Items in the cart that can't be sold in the visitor's country."""
        return [product_map[pid] for pid in cart() if pid in product_map and not price(product_map[pid])["available"]]

    @app.context_processor
    def inject_globals():
        return {"cfg": app.config, "model_info": classifier.info(),
                "cart_count": sum(session.get("cart", {}).values()), "pest_names": names,
                "verdicts": VERDICTS, "model_labels": set(classifier.labels), "is_admin": is_admin(),
                "price": price, "country": current_country(), "countries": COUNTRIES, "home": HOME}

    @app.template_filter("money")
    def money(value):
        return "Free" if not value else f"{app.config['CURRENCY']} {value:,.2f}"

    @app.template_filter("cedis")
    def cedis(value):
        return f"{app.config['CURRENCY']} {value or 0:,.2f}"

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        flash("Image is too large (max 12 MB).", "error")
        return redirect(url_for("scan"))

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("404.html"), 404

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", title="Request not accepted",
                               message=getattr(e, "description", "") or "Please go back and try again."), 400

    @app.errorhandler(429)
    def too_many(_e):
        return render_template("error.html", title="Too many requests",
                               message="You're doing that too often. Please wait a few minutes and try again."), 429

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error happened. Please try again."), 500

    # --------------------------------------------------------------- pages
    @app.route("/")
    def index():
        featured = [p for p in products if p["category"] != "App Plans"][:4]
        plans = [p for p in products if p["category"] == "App Plans"]
        return render_template("index.html", pests=pests, featured=featured, plans=plans,
                               stats=db.scan_stats())  # only the total count is shown

    @app.route("/scan", methods=["GET", "POST"])
    @rate_limit("scan", 30, 600)
    def scan():
        if request.method == "POST":
            try:
                raw, source = read_submitted_image()
                scan_id, _ = run_detection(raw, source, request.form.get("crop", "").strip()[:60],
                                           request.form.get("location", "").strip()[:80],
                                           request.form.get("notes", "").strip()[:500],
                                           request.form.get("sprayed", "no"))
                return redirect(url_for("result", scan_id=scan_id))
            except ValueError as exc:
                flash(str(exc), "error")
        return render_template("scan.html")

    @app.route("/result/<int:scan_id>")
    def result(scan_id):
        record = own_scan_or_404(scan_id)
        r = record["result"]
        verified = record.get("verified_label")
        if verified in pests:
            # A person confirmed the condition: show its profile and safety verdict instead of the guess.
            info = pests[verified]
            r = dict(r, category=info["category"], uncertain=False,
                     edibility=assess(info, False, r.get("edibility", {}).get("sprayed", "no"), r.get("is_demo")))
        else:
            info = pest_info(record["label"])
        recommended = [product_map[p] for p in info.get("products", []) if p in product_map]
        return render_template("result.html", scan=record, r=r, pest=info, recommended=recommended,
                               slug=verified if verified in pests else record["label"])

    @app.route("/history")
    def history():
        label = request.args.get("label") or None
        return render_template("history.html", scans=db.list_scans(label, owner=scope()),
                               stats=db.scan_stats(owner=scope()), active=label)

    @app.post("/history/<int:scan_id>/delete")
    def delete_scan(scan_id):
        own_scan_or_404(scan_id)
        row = db.delete_scan(scan_id)
        if row:
            for f in row.values():
                if f:
                    try:
                        os.remove(os.path.join(app.config["UPLOAD_DIR"], f))
                    except OSError:
                        pass
            flash("Scan deleted.", "success")
        return redirect(url_for("history"))

    # -------------------------------------------------------------- gallery
    CROPS = ["Tomato", "Maize", "Cassava", "Cocoa", "Pepper", "Cabbage", "Cowpea", "Okra", "Onion",
             "Garden egg", "Groundnut", "Rice", "Sorghum", "Yam", "Plantain", "Other"]
    app.config.setdefault("CROPS", CROPS)

    @app.route("/gallery")
    def gallery():
        crop = request.args.get("crop") or None
        label = request.args.get("label") or None
        eat = request.args.get("eat") or None
        return render_template("gallery.html", items=db.gallery_items(crop, label, eat, owner=scope()),
                               counts=db.gallery_counts(owner=scope()), crop=crop, label=label, eat=eat)

    @app.post("/gallery/upload")
    @rate_limit("bulk", 10, 600)
    def gallery_upload():
        """Bulk upload: analyse and store several photos of one crop at once."""
        files = [f for f in request.files.getlist("images") if f and f.filename][:30]
        crop = request.form.get("crop", "").strip()[:60]
        if not files:
            flash("Choose one or more images to upload.", "error")
            return redirect(url_for("gallery"))
        ok, bad = 0, []
        for f in files:
            if not allowed(f.filename):
                bad.append(f.filename)
                continue
            try:
                run_detection(f.read(), "upload", crop, request.form.get("location", "").strip()[:80],
                              sprayed=request.form.get("sprayed", "no"))
                ok += 1
            except ValueError:
                bad.append(f.filename)
        flash(f"Saved and analysed {ok} image{'s' if ok != 1 else ''}"
              + (f"; skipped {len(bad)} invalid file(s)." if bad else "."), "success" if ok else "error")
        return redirect(url_for("gallery", crop=crop or None))

    @app.post("/result/<int:scan_id>/verify")
    @rate_limit("verify", 60, 600)
    def verify_scan(scan_id):
        own_scan_or_404(scan_id)
        label = request.form.get("verified_label", "")
        if label not in pests and label != "other":
            abort(400)
        sprayed = db.get_scan(scan_id)["result"].get("edibility", {}).get("sprayed", "no")
        eat = assess(pests[label], False, sprayed)["status"] if label in pests else "unknown"
        db.set_verified_label(scan_id, label, eat)
        flash("Thanks! The photo is now labelled in the gallery.", "success")
        return redirect(url_for("result", scan_id=scan_id))

    @app.get("/gallery/export.zip")
    @admin_required
    def gallery_export():
        """Download stored images as a training dataset: one folder per label + metadata.csv.

        ?verified=1 exports only photos a person confirmed (recommended for retraining).
        """
        verified = request.args.get("verified") == "1"
        items = db.gallery_items(request.args.get("crop") or None, verified_only=verified, limit=100000)
        # skip "other" and unconfirmed low-confidence guesses - they would be wrong training labels
        items = [i for i in items if i["final_label"] != "other"
                 and (i["verified_label"] or i["category"] != "unrecognised")]
        if not items:
            flash("No images to export yet" + (" (confirm some labels first)." if verified else "."), "error")
            return redirect(url_for("gallery"))
        tmp = tempfile.TemporaryFile()
        meta = io.StringIO()
        writer = csv.writer(meta)
        writer.writerow(["file", "label", "predicted_label", "confidence", "verified", "crop", "location",
                         "eat_status", "created_at"])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in items:
                src = os.path.join(app.config["UPLOAD_DIR"], i["image_file"])
                if not os.path.exists(src):
                    continue
                arc = f"{i['final_label']}/{i['id']:06d}_{i['image_file']}"
                zf.write(src, arc)
                writer.writerow([csv_safe(v) for v in (
                    arc, i["final_label"], i["label"], round(i["confidence"], 4), int(i["verified_label"] is not None),
                    i["crop"] or "", i["location"] or "", i["eat_status"] or "", i["created_at"])])
            zf.writestr("metadata.csv", meta.getvalue())
        tmp.seek(0)
        name = "fdcs_dataset_verified.zip" if verified else "fdcs_crop_images.zip"
        return send_file(tmp, mimetype="application/zip", as_attachment=True, download_name=name)

    @app.route("/pests")
    def pest_library():
        q = request.args.get("q", "").strip().lower()
        cat = request.args.get("category")
        items = {k: v for k, v in pests.items()
                 if (not q or q in (v["common_name"] + v["scientific_name"] + v.get("group", "")
                                    + " ".join(v["crops"])).lower())
                 and (not cat or v.get("category") == cat)}
        return render_template("pests.html", pests=items, q=q, active=cat)

    @app.route("/pests/<slug>")
    def pest_detail(slug):
        if slug not in pests:
            abort(404)
        info = pests[slug]
        recommended = [product_map[p] for p in info.get("products", []) if p in product_map]
        return render_template("pest_detail.html", slug=slug, pest=info, recommended=recommended)

    @app.route("/about")
    def about():
        return render_template("about.html", pests=pests)

    @app.route("/contact", methods=["GET", "POST"])
    @rate_limit("contact", 5, 600)
    def contact():
        if request.method == "POST":
            name, email = request.form.get("name", "").strip(), request.form.get("email", "").strip()
            message = request.form.get("message", "").strip()
            if not name or not EMAIL_RE.match(email) or len(message) < 5:
                flash("Please fill in your name, a valid email and a message.", "error")
            else:
                db.add_message(name[:100], email[:120], message[:2000])
                flash("Thanks! Your message has been received.", "success")
                return redirect(url_for("contact"))
        return render_template("contact.html")

    # --------------------------------------------------------------- store
    @app.route("/store")
    def store():
        category = request.args.get("category")
        categories = sorted({p["category"] for p in products}, key=lambda c: (c != "App Plans", c))
        items = [p for p in products if not category or p["category"] == category]
        return render_template("store.html", products=items, categories=categories, active=category)

    @app.route("/store/<pid>")
    def product(pid):
        item = product_map.get(pid) or abort(404)
        helps = [(k, v) for k, v in pests.items() if pid in v.get("products", [])]
        return render_template("product.html", p=item, helps=helps)

    @app.post("/cart/add/<pid>")
    @rate_limit("cart", 120, 600)
    def cart_add(pid):
        if pid not in product_map:
            abort(404)
        q = price(product_map[pid])
        if not q["available"]:
            flash(f"{product_map[pid]['name']} is delivered in Ghana only. App plans can be bought from anywhere.",
                  "error")
            return redirect(safe_next(request.form.get("next"), url_for("store")))
        qty = max(1, min(99, request.form.get("qty", 1, type=int) or 1))
        items = cart()
        items[pid] = min(product_map[pid]["stock"], items.get(pid, 0) + qty)
        session.modified = True
        flash(f"Added {product_map[pid]['name']} to your cart.", "success")
        return redirect(safe_next(request.form.get("next"), url_for("view_cart")))

    @app.post("/cart/update")
    def cart_update():
        items = cart()
        for pid in list(items):
            qty = request.form.get(f"qty_{pid}", type=int)
            if qty is None or pid not in product_map:
                continue
            if qty <= 0:
                items.pop(pid)
            else:
                items[pid] = min(qty, product_map[pid]["stock"])
        session.modified = True
        return redirect(url_for("view_cart"))

    @app.route("/cart")
    def view_cart():
        lines, total, display_total = cart_lines()
        return render_template("cart.html", lines=lines, total=total, display_total=display_total,
                               blocked=cart_blocked())

    @app.post("/country")
    def set_country():
        code = request.form.get("country", "")
        if code in COUNTRIES:
            session["country"] = code
            blocked = cart_blocked()
            for p in blocked:
                cart().pop(p["id"], None)
            session.modified = True
            if blocked:
                flash("Removed items that are delivered in Ghana only.", "error")
        return redirect(safe_next(request.form.get("next"), url_for("store")))

    # ------------------------------------------------------------ checkout
    def paystack():
        key = app.config.get("PAYSTACK_SECRET_KEY")
        return app.extensions.get("fdcs_paystack") or (Paystack(key) if key else None)

    def order_url(order, external=False):
        return url_for("order_confirmation", order_id=order["id"], token=order["token"], _external=external)

    @app.route("/checkout", methods=["GET", "POST"])
    @rate_limit("checkout", 10, 600)
    def checkout():
        lines, total, display_total = cart_lines()
        if not lines:
            flash("Your cart is empty.", "error")
            return redirect(url_for("store"))
        country = current_country()
        online = paystack() is not None
        # Cash on delivery only in Ghana; abroad, pay online or send MoMo (e.g. via a remittance app).
        methods = (["paystack"] if online else []) + ["momo"] + (["cod"] if country == HOME else [])
        ctx = dict(lines=lines, total=total, display_total=display_total, methods=methods)
        form = request.form
        if request.method == "POST":
            customer = {k: form.get(k, "").strip() for k in
                        ("name", "phone", "email", "address", "payment_method")}
            errors = []
            if len(customer["name"]) < 2:
                errors.append("Enter your full name.")
            if not PHONE_RE.match(customer["phone"]):
                errors.append("Enter a valid phone number.")
            if customer["email"] and not EMAIL_RE.match(customer["email"]):
                errors.append("Enter a valid email or leave it blank.")
            if customer["payment_method"] == "paystack" and not customer["email"]:
                errors.append("Paystack needs an email address for your receipt.")
            if len(customer["address"]) < 3:
                errors.append("Enter a delivery address or digital address (GhanaPost GPS).")
            if total > 0 and customer["payment_method"] not in methods:
                errors.append("Choose a payment method.")
            if errors:
                for e in errors:
                    flash(e, "error")
                return render_template("checkout.html", form=form, **ctx)

            items = [{"id": l["product"]["id"], "name": l["product"]["name"],
                      "price": round(l["subtotal"] / l["qty"], 2), "qty": l["qty"], "subtotal": l["subtotal"],
                      "shown_as": l["unit"]}
                     for l in lines]
            if total == 0:
                customer["payment_method"], status = "free", "confirmed"
            else:
                status = {"cod": "cash_on_delivery"}.get(customer["payment_method"], "awaiting_payment")
            token = secrets.token_urlsafe(12)
            order_id = db.add_order(customer, items, total, status, token, country, display_total)
            reference = new_reference(order_id)
            db.update_order(order_id, payment_ref=reference)
            session["cart"] = {}
            order = db.get_order(order_id)
            if customer["payment_method"] == "paystack":
                try:
                    return redirect(paystack().initialize(
                        customer["email"], total, reference,
                        url_for("paystack_callback", _external=True),
                        {"order_id": order_id, "phone": customer["phone"]}))
                except PaymentError as exc:
                    flash(f"{exc} Your order is saved - you can pay by direct Mobile Money transfer below.", "error")
                    db.update_order(order_id, admin_note="Paystack start failed; offered direct MoMo")
            return redirect(order_url(order))
        return render_template("checkout.html", form=form, **ctx)

    @app.route("/order/<int:order_id>/<token>", methods=["GET", "POST"])
    @rate_limit("order", 10, 600)
    def order_confirmation(order_id, token):
        order = db.get_order(order_id)
        if order is None or not order.get("token") or not secrets.compare_digest(order["token"], token):
            abort(404)
        if request.method == "POST" and order["status"] in ("awaiting_payment", "payment_submitted"):
            txn = re.sub(r"[^A-Za-z0-9.\-]", "", request.form.get("momo_txn_id", ""))[:40]
            sender = request.form.get("momo_sender", "").strip()
            if len(txn) < 4 or not PHONE_RE.match(sender):
                flash("Enter the transaction ID from your MoMo SMS and the number you paid from.", "error")
            else:
                db.update_order(order_id, momo_txn_id=txn, momo_sender=sender, status="payment_submitted")
                flash("Thank you! We'll confirm your payment and contact you shortly.", "success")
                return redirect(order_url(order))
        return render_template("order.html", order=db.get_order(order_id))

    @app.get("/payment/paystack/callback")
    def paystack_callback():
        """Customer returns here from Paystack. We verify with Paystack's server before marking paid."""
        order = db.find_order_by_ref(request.args.get("reference", ""))
        if order is None:
            abort(404)
        if order["status"] not in ("paid", "delivered") and paystack():
            try:
                result = paystack().verify(order["payment_ref"], order["total"])
                if result["paid"]:
                    db.update_order(order["id"], status="paid", paid_at=result["paid_at"] or db.now())
                    flash("Payment received - thank you!", "success")
                else:
                    flash(f"Payment not completed (status: {result['status']}). You can try again "
                          "or pay by direct Mobile Money transfer.", "error")
            except PaymentError as exc:
                flash(str(exc), "error")
        return redirect(order_url(order))

    @app.post("/payment/paystack/webhook")
    def paystack_webhook():
        """Paystack calls this server-to-server when a payment succeeds (set the URL in your dashboard)."""
        ps = paystack()
        if ps is None or not ps.valid_webhook(request.get_data(), request.headers.get("x-paystack-signature", "")):
            abort(401)
        event = request.get_json(silent=True) or {}
        if event.get("event") == "charge.success":
            order = db.find_order_by_ref(event.get("data", {}).get("reference", ""))
            if order and order["status"] not in ("paid", "delivered"):
                result = ps.verify(order["payment_ref"], order["total"])
                if result["paid"]:
                    db.update_order(order["id"], status="paid", paid_at=result["paid_at"] or db.now())
        return "", 200

    # --------------------------------------------------------------- admin
    @app.route("/admin/login", methods=["GET", "POST"])
    @rate_limit("login", 5, 900)  # 5 attempts per 15 minutes per IP stops password guessing
    def admin_login():
        configured = bool(app.config.get("ADMIN_PASSWORD"))
        if request.method == "POST" and configured:
            sent = request.form.get("password", "").encode()
            if secrets.compare_digest(sent, app.config["ADMIN_PASSWORD"].encode()):
                keep = {k: session[k] for k in ("vid", "cart") if k in session}
                session.clear()  # new session on login (prevents session fixation)
                session.update(keep, is_admin=True, admin_at=time.time())
                session.permanent = True
                app.logger.info("Admin login from %s", request.remote_addr)
                nxt = safe_next(request.args.get("next"), url_for("admin_orders"))
                return redirect(nxt if nxt.startswith("/admin") or nxt.startswith("/gallery") else url_for("admin_orders"))
            app.logger.warning("Failed admin login from %s", request.remote_addr)
            flash("Wrong password.", "error")
        return render_template("admin_login.html", configured=configured)

    @app.get("/admin/logout")
    def admin_logout():
        session.clear()
        return redirect(url_for("index"))

    @app.get("/admin")
    @admin_required
    def admin_orders():
        status = request.args.get("status") or None
        return render_template("admin_orders.html", orders=db.list_orders(status), totals=db.order_totals(),
                               active=status, messages=db.list_messages()[:20],
                               online=paystack() is not None)

    @app.post("/admin/orders/<int:order_id>")
    @admin_required
    def admin_update_order(order_id):
        order = db.get_order(order_id) or abort(404)
        action = request.form.get("action")
        changes = {"paid": {"status": "paid", "paid_at": db.now()}, "delivered": {"status": "delivered"},
                   "cancelled": {"status": "cancelled"},
                   "unpaid": {"status": "awaiting_payment", "paid_at": None}}.get(action)
        if changes is None:
            abort(400)
        note = request.form.get("note", "").strip()[:200]
        if note:
            changes["admin_note"] = note
        db.update_order(order["id"], **changes)
        flash(f"Order #{order_id} marked {changes['status'].replace('_', ' ')}.", "success")
        return redirect(url_for("admin_orders", status=request.form.get("return_status") or None))

    @app.route("/admin/prices", methods=["GET", "POST"])
    @admin_required
    def admin_prices():
        """Owner edits prices to follow the market. Ghana price in GH₵; app plans also get a US$ price abroad."""
        if request.method == "POST":
            if request.form.get("action") == "refresh_rates":
                ok = rates.refresh(force=True)
                flash("Exchange rates updated." if ok else "Couldn't reach the exchange-rate service; "
                      "still using the last saved rates.", "success" if ok else "error")
                return redirect(url_for("admin_prices"))
            changed, errors = 0, []
            for pid, p in product_map.items():
                raw = request.form.get(f"price_{pid}", "").strip()
                raw_usd = request.form.get(f"usd_{pid}", "").strip()
                try:
                    new_price = round(float(raw), 2)
                    new_usd = round(float(raw_usd), 2) if p["category"] == "App Plans" else None
                except ValueError:
                    errors.append(p["name"])
                    continue
                if not (0 <= new_price <= 1_000_000) or (new_usd is not None and not 0 <= new_usd <= 100_000):
                    errors.append(p["name"])
                    continue
                if new_price != p["price"] or (new_usd is not None and new_usd != p.get("price_usd_intl")):
                    db.set_price(pid, new_price, new_usd)
                    changed += 1
            apply_price_overrides()
            if errors:
                flash("Check the prices for: " + ", ".join(errors), "error")
            flash(f"Saved {changed} price change{'s' if changed != 1 else ''}.", "success")
            return redirect(url_for("admin_prices"))
        return render_template("admin_prices.html", products=products, rates=rates,
                               ghs_per_usd=rates.rates.get("GHS"))

    # ----------------------------------------------------------------- API
    @app.post("/api/predict")
    @rate_limit("api", 60, 600)
    def api_predict():
        """JSON endpoint for the Android app or other clients. Send multipart 'image'."""
        key = app.config.get("API_KEY")
        if key and not secrets.compare_digest(request.headers.get("X-API-Key", "").encode(), key.encode()):
            return jsonify(error="Missing or wrong X-API-Key header."), 401
        file = request.files.get("image")
        if not file or not file.filename:
            return jsonify(error="Send an image file in the 'image' field."), 400
        try:
            scan_id, data = run_detection(file.read(), "api", request.form.get("crop", "")[:60],
                                          sprayed=request.form.get("sprayed", "no"), owner="api")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        info = pest_info(data["label"])
        return jsonify(scan_id=scan_id, result=data, safe_to_eat=data["edibility"],
            organism=data["organism"], category=data["category"],
            management=info["management"],
            annotated_url=url_for("uploaded", filename=db.get_scan(scan_id)["annotated_file"],
                                  _external=True))

    @app.get("/api/pests")
    def api_pests():
        return jsonify(pests)

    @app.get("/api/health")
    def api_health():
        return jsonify(status="ok", model=classifier.info())

    # ------------------------------------------------------ installable web app
    @app.get("/manifest.webmanifest")
    def manifest():
        data = {
            "name": f"{app.config['APP_NAME']} Crop Health", "short_name": app.config["APP_NAME"],
            "description": "Photograph a crop to find the pest or disease, the organism behind it, "
                           "and whether the produce is safe to eat.",
            "id": "/", "start_url": "/?source=app", "scope": "/", "display": "standalone",
            "orientation": "portrait", "background_color": "#fbfcfb", "theme_color": "#1b6b3a",
            "lang": "en", "categories": ["agriculture", "productivity", "shopping"],
            "icons": [
                {"src": url_for("static", filename="icons/icon-192.png"), "sizes": "192x192", "type": "image/png"},
                {"src": url_for("static", filename="icons/icon-512.png"), "sizes": "512x512", "type": "image/png"},
                {"src": url_for("static", filename="icons/icon-maskable-512.png"), "sizes": "512x512",
                 "type": "image/png", "purpose": "maskable"}],
            "shortcuts": [
                {"name": "Check a crop", "url": "/scan", "icons": [
                    {"src": url_for("static", filename="icons/icon-192.png"), "sizes": "192x192"}]},
                {"name": "Store", "url": "/store"}],
        }
        resp = app.response_class(json.dumps(data), mimetype="application/manifest+json")
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp

    @app.get("/sw.js")
    def service_worker():
        """Served from the site root so it can work for every page."""
        resp = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript", max_age=0)
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Service-Worker-Allowed"] = "/"
        return resp

    @app.get("/offline")
    def offline():
        return render_template("offline.html")

    @app.route("/uploads/<filename>")
    def uploaded(filename):
        # Only our own random image names (e.g. 3f2a...e1_det.jpg) - no paths, no other file types.
        if not re.fullmatch(r"[0-9a-f]{32}(_det)?\.jpg", filename):
            abort(404)
        return send_from_directory(app.config["UPLOAD_DIR"], filename, max_age=86400)

    return app


app = create_app()

if __name__ == "__main__":
    if os.environ.get("FLASK_DEBUG") == "1":  # developer mode with auto-reload
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
    else:  # same production server as `python serve.py`, on port 8000
        import serve
        serve.main(app)
