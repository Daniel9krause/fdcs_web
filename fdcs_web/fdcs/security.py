"""Security layer for the FDCS web app.

- CSRF protection on every form (token in session + hidden field / header)
- Security headers: Content-Security-Policy, clickjacking, MIME sniffing, HSTS on HTTPS
- Hardened session cookies and a persistent random SECRET_KEY
- Rate limiting (login brute force, upload flooding, checkout spam)
- Safe redirects (no open-redirect)
- Per-browser visitor id so people only see their own scans
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps

from flask import abort, current_app, g, request, session
from PIL import Image

# Refuse "decompression bomb" images (tiny files that expand to gigantic pixel counts).
Image.MAX_IMAGE_PIXELS = 40_000_000

CSRF_EXEMPT = {"paystack_webhook", "api_predict", "static", "uploaded"}


# ---------------------------------------------------------------- secret key
def load_secret_key(configured: str, instance_dir: str) -> str:
    """Use FDCS_SECRET_KEY if set; otherwise create a random key once and keep it in instance/."""
    if configured and configured != "change-me-in-production":
        return configured
    os.makedirs(instance_dir, exist_ok=True)
    path = os.path.join(instance_dir, "secret_key")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
        if len(key) >= 32:
            return key
    key = secrets.token_hex(32)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


# ---------------------------------------------------------------- rate limits
class RateLimiter:
    """Small in-memory sliding-window limiter (per process). Good enough for one server."""

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: int) -> bool:
        """Record a hit; return False if the key is over `limit` hits in `window` seconds."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 50_000:  # keep memory bounded
                self._hits.clear()
            return True

    def reset(self, key: str):
        with self._lock:
            self._hits.pop(key, None)


limiter = RateLimiter()


def client_ip() -> str:
    return request.remote_addr or "unknown"


def rate_limit(name: str, limit: int, window: int):
    """Decorator: at most `limit` POST requests per IP per `window` seconds."""
    def deco(view):
        @wraps(view)
        def wrapper(*a, **kw):
            if request.method == "POST" and not current_app.config.get("RATE_LIMITS_DISABLED"):
                if not limiter.hit(f"{name}:{client_ip()}", limit, window):
                    abort(429)
            return view(*a, **kw)
        return wrapper
    return deco


# ---------------------------------------------------------------- CSRF
def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def check_csrf():
    if request.method in ("GET", "HEAD", "OPTIONS") or request.endpoint in CSRF_EXEMPT:
        return
    if current_app.config.get("CSRF_DISABLED"):
        return
    sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token", "")
    expected = session.get("_csrf", "")
    if not expected or not sent or not secrets.compare_digest(sent, expected):
        abort(400, description="Your session expired or the form was not sent from this site. "
                               "Please go back, refresh the page and try again.")


# ---------------------------------------------------------------- visitor id
def visitor_id() -> str:
    """Random id for this browser, stored in the signed session cookie."""
    if "vid" not in session:
        session["vid"] = secrets.token_urlsafe(16)
        session.permanent = True
    return session["vid"]


# ---------------------------------------------------------------- redirects
def safe_next(target: str | None, fallback: str) -> str:
    """Only allow redirects to paths on this site (blocks //evil.com and https://evil.com)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return fallback


# ---------------------------------------------------------------- headers
def security_headers(response):
    csp = ("default-src 'self'; "
           "script-src 'self'; "
           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
           "font-src 'self' https://fonts.gstatic.com; "
           "img-src 'self' data: blob:; "
           "media-src 'self' blob:; "
           "connect-src 'self'; "
           "frame-ancestors 'none'; base-uri 'self'; object-src 'none'; "
           "form-action 'self' https://checkout.paystack.com")
    h = response.headers
    h.setdefault("Content-Security-Policy", csp)
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    h.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=(), payment=()")
    h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if request.is_secure:
        h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if request.endpoint in ("admin_orders", "admin_login", "admin_prices", "order_confirmation", "checkout",
                            "result") or session.get("is_admin"):
        h["Cache-Control"] = "no-store"  # never kept by the browser or the offline cache
    h.pop("Server", None)
    return response


def init_security(app):
    instance = os.path.join(app.root_path, "instance")
    app.config["SECRET_KEY"] = load_secret_key(app.config.get("SECRET_KEY", ""), instance)
    https = app.config.get("HTTPS", False)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=https,
        SESSION_COOKIE_NAME="fdcs_session",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
    )
    if app.config.get("BEHIND_PROXY"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.before_request(check_csrf)
    app.after_request(security_headers)
    app.jinja_env.globals["csrf_token"] = csrf_token
