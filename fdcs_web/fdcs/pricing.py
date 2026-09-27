"""Country-aware pricing with live exchange rates.

- Customers in Ghana pay the Ghana price in cedis (GH₵).
- Customers abroad see app plans at their international US$ price (e.g. US$35), converted to
  their own currency at today's exchange rate. Physical supplies are delivered in Ghana only.
- Every order is charged in GH₵ (Mobile Money and Paystack Ghana settle in cedis), so abroad the
  US$ price is converted to cedis at today's rate, and the customer sees both amounts.

Rates come from open.er-api.com (free, no key), refreshed at most every 12 hours and cached in
instance/rates.json. If the internet is down, the last saved rates (or built-in ones) are used.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

RATES_URL = "https://open.er-api.com/v6/latest/USD"
REFRESH_SECONDS = 12 * 3600
RETRY_SECONDS = 3600

# Built-in fallback: units of each currency per 1 US$ (open.er-api.com, 26 Sep 2026).
FALLBACK_RATES = {"USD": 1.0, "GHS": 11.567841, "NGN": 1328.960147, "XOF": 575.453073, "XAF": 575.453073,
                  "KES": 129.511518, "UGX": 3833.29746, "TZS": 2645.139666, "ZAR": 16.304692,
                  "EGP": 51.682464, "GBP": 0.755351, "EUR": 0.877267, "CAD": 1.413327, "AUD": 1.423404,
                  "INR": 95.918696, "CNY": 6.724641}
FALLBACK_DATE = "Sat, 26 Sep 2026 00:02:32 +0000"

# code: (name, currency, symbol, decimals)
COUNTRIES = {
    "GH": ("Ghana", "GHS", "GH₵", 2),
    "NG": ("Nigeria", "NGN", "₦", 0),
    "CI": ("Côte d'Ivoire", "XOF", "CFA ", 0),
    "TG": ("Togo", "XOF", "CFA ", 0),
    "BF": ("Burkina Faso", "XOF", "CFA ", 0),
    "BJ": ("Benin", "XOF", "CFA ", 0),
    "SN": ("Senegal", "XOF", "CFA ", 0),
    "CM": ("Cameroon", "XAF", "FCFA ", 0),
    "KE": ("Kenya", "KES", "KSh ", 0),
    "UG": ("Uganda", "UGX", "USh ", 0),
    "TZ": ("Tanzania", "TZS", "TSh ", 0),
    "ZA": ("South Africa", "ZAR", "R ", 2),
    "EG": ("Egypt", "EGP", "E£ ", 2),
    "GB": ("United Kingdom", "GBP", "£", 2),
    "DE": ("Germany", "EUR", "€", 2),
    "FR": ("France", "EUR", "€", 2),
    "NL": ("Netherlands", "EUR", "€", 2),
    "US": ("United States", "USD", "US$", 2),
    "CA": ("Canada", "CAD", "CA$", 2),
    "AU": ("Australia", "AUD", "A$", 2),
    "IN": ("India", "INR", "₹", 0),
    "CN": ("China", "CNY", "¥", 2),
    "XX": ("Other country", "USD", "US$", 2),
}
HOME = "GH"


def fmt(amount: float, currency_symbol: str, decimals: int) -> str:
    return f"{currency_symbol}{amount:,.{decimals}f}"


class Rates:
    """Thread-safe exchange-rate cache (units per 1 US$)."""

    def __init__(self, cache_path: str, fetch=None, auto_refresh=True):
        self.cache_path = cache_path
        self._fetch = fetch or self._download
        self.auto_refresh = auto_refresh
        self._lock = threading.Lock()
        self.rates, self.updated, self.source = dict(FALLBACK_RATES), FALLBACK_DATE, "built-in"
        self._checked = 0.0
        self._load_cache()

    @staticmethod
    def _download() -> dict:
        req = urllib.request.Request(RATES_URL, headers={"User-Agent": "FDCS/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read().decode())

    def _load_cache(self):
        try:
            with open(self.cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("rates", {}).get("GHS"):
                self.rates = {**FALLBACK_RATES, **data["rates"]}
                self.updated, self.source = data["updated"], "saved"
                self._checked = data.get("fetched_at", 0)
        except (OSError, ValueError, KeyError):
            pass

    def refresh(self, force=False) -> bool:
        """Download fresh rates if due. Returns True on success. Never raises."""
        now = time.time()
        with self._lock:
            if not force and now - self._checked < REFRESH_SECONDS:
                return False
            self._checked = now - REFRESH_SECONDS + RETRY_SECONDS  # if it fails, retry in an hour
        try:
            data = self._fetch()
            rates = data["rates"]
            if not (data.get("result") == "success" and 1 < float(rates["GHS"]) < 1000):
                return False
        except Exception:
            return False
        with self._lock:
            self.rates = {**FALLBACK_RATES, **{k: float(v) for k, v in rates.items()}}
            self.updated, self.source, self._checked = data.get("time_last_update_utc", ""), "live", now
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as fh:
                    json.dump({"rates": self.rates, "updated": self.updated, "fetched_at": now}, fh)
            except OSError:
                pass
        return True

    def refresh_in_background(self):
        if time.time() - self._checked >= REFRESH_SECONDS:
            threading.Thread(target=self.refresh, daemon=True).start()

    def per_usd(self, currency: str) -> float:
        if self.auto_refresh:
            self.refresh_in_background()
        return self.rates.get(currency, 1.0 if currency == "USD" else FALLBACK_RATES.get(currency, 1.0))


def country_from_accept_language(header: str) -> str:
    """'en-NG,en;q=0.9' -> 'NG' if we support it, else Ghana."""
    for part in (header or "").split(","):
        tag = part.split(";")[0].strip()
        if "-" in tag:
            code = tag.split("-")[-1].upper()
            if code in COUNTRIES:
                return code
    return HOME


def quote(product: dict, country: str, rates: Rates) -> dict:
    """Price of one unit for a customer in `country`.

    Returns: available, charge_ghs (what is actually charged), display (in their currency),
    and usd (the international price) where relevant.
    """
    name, cur, sym, dec = COUNTRIES.get(country, COUNTRIES["XX"])
    if country == HOME:
        return {"available": True, "charge_ghs": float(product["price"]),
                "display": fmt(product["price"], "GH₵ ", 2) if product["price"] else "Free",
                "value": float(product["price"]), "currency": "GHS", "usd": None, "reason": ""}
    if product.get("category") != "App Plans":
        return {"available": False, "charge_ghs": 0.0, "display": fmt(product["price"], "GH₵ ", 2),
                "value": 0.0, "currency": cur, "usd": None, "reason": "Delivered in Ghana only"}
    usd = float(product.get("price_usd_intl", 0) or 0)
    if usd == 0:
        return {"available": True, "charge_ghs": 0.0, "display": "Free", "value": 0.0, "currency": cur,
                "usd": 0.0, "reason": ""}
    local = usd * rates.per_usd(cur)
    return {"available": True, "charge_ghs": round(usd * rates.per_usd("GHS"), 2), "display": fmt(local, sym, dec),
            "value": round(local, dec), "currency": cur, "usd": usd, "reason": ""}
