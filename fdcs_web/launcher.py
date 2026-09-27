"""Start FDCS with one command (used by start_public.bat and start_local.bat).

    python launcher.py public    # site + free Cloudflare HTTPS link that works on any phone
    python launcher.py local     # site on this computer and the same Wi-Fi only

Public mode:
  1. asks for the owner password (hidden) if FDCS_ADMIN_PASSWORD isn't set,
  2. starts the web server,
  3. downloads Cloudflare's `cloudflared` the first time (about 60 MB, free, no account),
  4. opens a secure https://....trycloudflare.com link and shows it with a QR code to scan.
The link works as long as this window stays open and the computer stays awake.
"""
from __future__ import annotations

import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, "tools")
INSTANCE = os.path.join(ROOT, "instance")
PORT = int(os.environ.get("PORT", 8000))
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

DOWNLOADS = {
    ("Windows", "AMD64"): "cloudflared-windows-amd64.exe",
    ("Windows", "x86"): "cloudflared-windows-386.exe",
    ("Windows", "ARM64"): "cloudflared-windows-amd64.exe",
    ("Linux", "x86_64"): "cloudflared-linux-amd64",
    ("Linux", "aarch64"): "cloudflared-linux-arm64",
    ("Darwin", "x86_64"): "cloudflared-darwin-amd64.tgz",
    ("Darwin", "arm64"): "cloudflared-darwin-amd64.tgz",
}
RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def say(msg=""):
    print(msg, flush=True)


def find_cloudflared() -> str | None:
    exe = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    for path in (shutil.which("cloudflared"), os.path.join(TOOLS, exe), os.path.join(ROOT, exe),
                 os.path.join(os.path.dirname(ROOT), exe)):
        if path and os.path.isfile(path):
            return path
    return None


def download_cloudflared() -> str:
    name = DOWNLOADS.get((platform.system(), platform.machine()))
    if not name or name.endswith(".tgz"):
        raise SystemExit("Please install cloudflared yourself: "
                         "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    os.makedirs(TOOLS, exist_ok=True)
    target = os.path.join(TOOLS, "cloudflared.exe" if os.name == "nt" else "cloudflared")
    say(f"Downloading Cloudflare Tunnel (first time only, about 60 MB)...")
    tmp = target + ".part"
    with urllib.request.urlopen(RELEASE + name, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done, shown = 0, -1
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != shown and pct % 5 == 0:
                shown = pct
                print(f"\r  {pct:3d}%", end="", flush=True)
    os.replace(tmp, target)
    if os.name != "nt":
        os.chmod(target, 0o755)
    say("\r  done.   ")
    return target


def server_ready(timeout=120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(1)
    return False


def show_qr(url: str):
    try:
        import qrcode
    except ImportError:
        say("  (Tip: `pip install qrcode` to show a QR code here.)")
        return
    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)
    try:
        qr.print_ascii(invert=True)
    except (UnicodeEncodeError, OSError):
        pass
    os.makedirs(INSTANCE, exist_ok=True)
    png = os.path.join(INSTANCE, "public_link_qr.png")
    qr.make_image(fill_color="#0f3d2e", back_color="white").save(png)
    say(f"  QR code image saved: {png}")
    if os.name == "nt":
        try:
            os.startfile(png)  # opens the QR so a phone can scan it from the screen
        except OSError:
            pass


def ask_password(env):
    if env.get("FDCS_ADMIN_PASSWORD"):
        return
    say("Owner password for the /admin page (orders, payments, prices).")
    say("Press Enter to skip - the owner page will stay switched off.")
    pw = getpass.getpass("Password (hidden as you type): ")
    if pw:
        env["FDCS_ADMIN_PASSWORD"] = pw


def _stop(*_):
    raise KeyboardInterrupt


def main():
    import signal
    for sig in ("SIGTERM", "SIGBREAK", "SIGHUP"):  # window closed / system shutdown -> clean stop
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), _stop)
            except (ValueError, OSError):
                pass
    mode = (sys.argv[1] if len(sys.argv) > 1 else "public").lower()
    env = dict(os.environ, PORT=str(PORT), PYTHONUNBUFFERED="1")
    ask_password(env)
    if mode == "public":
        # Visitors arrive through Cloudflare: trust its forwarded address/https headers so rate limits
        # apply per visitor and links are built with https://
        env["FDCS_BEHIND_PROXY"] = "1"

    say("\nStarting the FDCS server (loading the AI model takes a few seconds)...")
    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve.py")], cwd=ROOT, env=env)
    procs = [server]
    try:
        if not server_ready():
            raise SystemExit("The server didn't start. Scroll up for the error message.")
        time.sleep(1)
        if server.poll() is not None:
            raise SystemExit(f"Port {PORT} is already in use - FDCS (or another program) is already running.\n"
                             "Close the other FDCS window first, then try again.")
        if mode != "public":
            webbrowser.open(f"http://127.0.0.1:{PORT}")
            server.wait()
            return

        cf = find_cloudflared() or download_cloudflared()
        say("Opening a secure public link through Cloudflare...")
        tunnel = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace")
        procs.append(tunnel)
        found = threading.Event()
        recent: list[str] = []

        def read_tunnel():
            for line in tunnel.stdout:
                recent.append(line.rstrip())
                del recent[:-15]
                m = URL_RE.search(line)
                if m and not found.is_set():
                    found.set()
                    url = m.group(0)
                    os.makedirs(INSTANCE, exist_ok=True)
                    with open(os.path.join(INSTANCE, "public_url.txt"), "w") as fh:
                        fh.write(url)
                    bar = "#" * 64
                    say(f"\n{bar}\n  YOUR PUBLIC WEBSITE (works on any phone, anywhere):\n\n    {url}\n")
                    say(f"  Owner page:  {url}/admin")
                    say("  Keep this window open. Closing it or sleeping the computer stops the link.")
                    say("  The address changes each time you start it again.")
                    say(bar)
                    show_qr(url)
                    say("\n  On the phone: open the link, then tap 'Install' (Android) or")
                    say("  Share > Add to Home Screen (iPhone) to get it as an app.\n")
                elif " ERR " in line or "error" in line.lower():
                    say("  [tunnel] " + line.strip())

        threading.Thread(target=read_tunnel, daemon=True).start()
        if not found.wait(60):
            say("\nCouldn't get a public link from Cloudflare. Last messages:")
            for line in recent:
                say("  " + line)
            say("Check the internet connection and try again. The site still works on this computer:"
                f" http://127.0.0.1:{PORT}")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        if tunnel.poll() is not None:
            say("\nThe Cloudflare tunnel stopped. Restart start_public.bat to get a new link.")
    except KeyboardInterrupt:
        say("\nStopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
