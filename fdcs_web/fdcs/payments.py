"""Payments: where customers' money goes.

Two ways to get paid, both ending up on the shop owner's Mobile Money wallet:

1. Direct MoMo transfer (works with no account): checkout shows the owner's MoMo
   number, the exact amount and an order reference. The customer sends the money
   from their own phone and types in the transaction ID; the owner confirms it on
   the admin page. Nothing is automatic, so the admin check is essential.

2. Paystack (automatic): when FDCS_PAYSTACK_SECRET_KEY is set, customers pay with a
   MoMo prompt or card on Paystack's secure page. Paystack verifies the payment and
   settles it to the owner's MoMo wallet or bank account chosen in the Paystack
   dashboard. The app marks the order paid only after verifying with Paystack.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import urllib.error
import urllib.request

PAYSTACK_API = "https://api.paystack.co"


class PaymentError(Exception):
    pass


def new_reference(order_id: int) -> str:
    return f"FDCS-{order_id}-{secrets.token_hex(3).upper()}"


def to_pesewas(amount_ghs: float) -> int:
    return int(round(amount_ghs * 100))


class Paystack:
    def __init__(self, secret_key: str, opener=None):
        self.secret_key = secret_key
        self._open = opener or urllib.request.urlopen  # injectable for tests

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            PAYSTACK_API + path, method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self.secret_key}", "Content-Type": "application/json"})
        try:
            with self._open(req, timeout=20) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read().decode()).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise PaymentError(f"Paystack error: {msg}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PaymentError("Could not reach Paystack. Check your internet connection.") from exc
        if not body.get("status"):
            raise PaymentError(body.get("message", "Paystack request failed"))
        return body["data"]

    def initialize(self, email: str, amount_ghs: float, reference: str, callback_url: str,
                   metadata: dict | None = None) -> str:
        """Start a payment; returns the Paystack checkout URL to redirect the customer to."""
        data = self._call("POST", "/transaction/initialize", {
            "email": email, "amount": to_pesewas(amount_ghs), "currency": "GHS",
            "reference": reference, "callback_url": callback_url,
            "channels": ["mobile_money", "card"], "metadata": metadata or {}})
        return data["authorization_url"]

    def verify(self, reference: str, expected_amount_ghs: float) -> dict:
        """Confirm with Paystack that this exact amount was paid. Never trust the browser."""
        data = self._call("GET", f"/transaction/verify/{reference}")
        ok = (data.get("status") == "success" and data.get("currency") == "GHS"
              and int(data.get("amount", 0)) == to_pesewas(expected_amount_ghs))
        return {"paid": ok, "status": data.get("status"), "channel": data.get("channel"),
                "paid_at": data.get("paid_at"), "amount": int(data.get("amount", 0)) / 100}

    def valid_webhook(self, raw_body: bytes, signature: str) -> bool:
        expected = hmac.new(self.secret_key.encode(), raw_body, hashlib.sha512).hexdigest()
        return hmac.compare_digest(expected, signature or "")
