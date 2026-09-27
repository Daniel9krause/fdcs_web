"""Entry point for Linux hosting (gunicorn, PythonAnywhere): `gunicorn wsgi:app`."""
from app import app  # noqa: F401
