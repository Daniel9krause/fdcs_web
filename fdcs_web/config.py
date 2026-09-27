"""Central configuration for the FDCS web application.

Every value can be overridden with an environment variable, so you can point the
app at a different model without editing code, e.g.

    FDCS_MODEL_PATH=models/my_model.tflite FDCS_NORMALIZE=mobilenet python app.py
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _path(env, default):
    value = os.environ.get(env, default)
    return value if os.path.isabs(value) else os.path.join(BASE_DIR, value)


class Config:
    APP_NAME = "FDCS"
    APP_FULL_NAME = "Farm_land Detect and Classification System"
    PROJECT_TITLE = ("Integration of CNN into a Mobile App for Pest Detection "
                     "and Classification on Farmland")
    CREATOR = "Daniel Krause"  # designed, built and trained the app
    # Edit this list to show your project team on the About page.
    TEAM = [
        {"name": "Daniel Krause", "role": "Creator: design, AI model, web app and store"},
    ]
    SUPERVISOR = os.environ.get("FDCS_SUPERVISOR", "")
    CURRENCY = "GH₵"
    CONTACT_EMAIL = os.environ.get("FDCS_CONTACT_EMAIL", "info@fdcs.example")
    CONTACT_PHONE = os.environ.get("FDCS_CONTACT_PHONE", "+233 00 000 0000")

    SECRET_KEY = os.environ.get("FDCS_SECRET_KEY", "change-me-in-production")

    # ---- Payments: all money goes to the shop owner --------------------------
    # Direct transfer: customers send Mobile Money to this number.
    MOMO_NUMBER = os.environ.get("FDCS_MOMO_NUMBER", "0200687324")
    MOMO_NETWORK = os.environ.get("FDCS_MOMO_NETWORK", "Telecel Cash")
    # Name registered on the wallet - customers see it on their phone before sending,
    # so showing it here lets them check they are paying the right person.
    MOMO_NAME = os.environ.get("FDCS_MOMO_NAME", "")
    # Automatic payments: set your Paystack secret key (sk_live_... or sk_test_...).
    # Choose MoMo 0200687324 as the settlement account in the Paystack dashboard.
    PAYSTACK_SECRET_KEY = os.environ.get("FDCS_PAYSTACK_SECRET_KEY", "")
    # Password for the owner's admin page (/admin) where orders and payments are confirmed.
    ADMIN_PASSWORD = os.environ.get("FDCS_ADMIN_PASSWORD", "")

    # ---- Security / hosting ------------------------------------------------
    HTTPS = os.environ.get("FDCS_HTTPS", "0") == "1"                # set to 1 when served over https
    BEHIND_PROXY = os.environ.get("FDCS_BEHIND_PROXY", "0") == "1"  # set to 1 on Render/Nginx/Cloudflare
    API_KEY = os.environ.get("FDCS_API_KEY", "")                     # if set, /api/predict needs X-API-Key
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12 MB uploads
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

    UPLOAD_DIR = _path("FDCS_UPLOAD_DIR", "uploads")
    DATABASE = _path("FDCS_DATABASE", "fdcs.sqlite3")
    PESTS_FILE = _path("FDCS_PESTS_FILE", "data/pests.json")
    PRODUCTS_FILE = _path("FDCS_PRODUCTS_FILE", "data/products.json")

    # ---- Model -----------------------------------------------------------
    # .tflite (preferred, same file as the Android app) or .keras / .h5
    MODEL_PATH = _path("FDCS_MODEL_PATH", "models/pest_model.tflite")
    LABELS_PATH = _path("FDCS_LABELS_PATH", "models/labels.txt")
    INPUT_SIZE = int(os.environ.get("FDCS_INPUT_SIZE", 224))
    # How pixels are scaled before inference:
    #   "none"      -> raw 0..255 floats (models from training/train.py include
    #                  their own preprocessing layer, so use this)
    #   "mobilenet" -> scaled to -1..1   (plain MobileNetV2 without the layer)
    #   "unit"      -> scaled to 0..1
    NORMALIZE = os.environ.get("FDCS_NORMALIZE", "none")
    HEALTHY_LABEL = os.environ.get("FDCS_HEALTHY_LABEL", "healthy")

    # ---- Detection -------------------------------------------------------
    TILE_GRID = int(os.environ.get("FDCS_TILE_GRID", 3))           # 3x3 overlapping tiles
    TILE_THRESHOLD = float(os.environ.get("FDCS_TILE_THRESHOLD", 0.55))
    MIN_CONFIDENCE = float(os.environ.get("FDCS_MIN_CONFIDENCE", 0.30))  # calibrated on test photos
