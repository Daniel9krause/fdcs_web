"""Production server. `python serve.py` and `python app.py` both start this.

    python serve.py                 # http://0.0.0.0:8000
    set PORT=5000 && python serve.py

It listens on every network card, so phones on the same Wi-Fi/hotspot can open it, and prints
the exact address to type on the phone. For access from any phone anywhere (with HTTPS, which
the live camera and "Install app" need), run start_public.bat instead.
"""
import logging
import os
import socket

from waitress import serve as waitress_serve


def lan_address():
    """Best guess at this computer's address on the local network (no data is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def main(app=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if app is None:
        from app import app  # imported here so the banner prints after the model loads

    port = int(os.environ.get("PORT", 8000))
    ip = lan_address()
    line = "=" * 64
    print(f"\n{line}\n  FDCS is running  (press Ctrl+C to stop)\n{line}")
    print(f"  On this computer:       http://127.0.0.1:{port}")
    if ip:
        print(f"  On phones (same Wi-Fi): http://{ip}:{port}")
        print("     - If phones can't open it, run allow_firewall.bat once (as administrator).")
        print("     - The live camera needs HTTPS; phones can still use Upload > Camera.")
    print("  For ANY phone, anywhere, with HTTPS: run start_public.bat")
    if not app.config["ADMIN_PASSWORD"]:
        print("  Owner page /admin is OFF - set FDCS_ADMIN_PASSWORD to turn it on.")
    else:
        print(f"  Owner page:             http://127.0.0.1:{port}/admin")
    print(line + "\n")
    waitress_serve(app, host="0.0.0.0", port=port, threads=8, max_request_body_size=14 * 1024 * 1024,
                   ident="FDCS", url_scheme="https" if app.config["HTTPS"] else "http")


if __name__ == "__main__":
    main()
