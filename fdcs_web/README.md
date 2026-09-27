# FDCS — Farm_land Detect and Classification System (Web)

A Python web application where a farmer **takes or uploads a photo of a crop** and gets four
answers from a **trained CNN (MobileNetV2, TensorFlow Lite)**:

1. **What's wrong with the crop?** — the pest or disease detected, with the affected areas boxed.
2. **Which organism causes it?** — its name and classification: insect, mite, fungus, water mould,
   bacterium or virus, with kingdom, order and family.
3. **Is the produce safe to eat?** — *Safe to eat*, *Eat with care*, *Not safe to eat*, or
   *Can't tell* — with what to cut away, wash or throw out.
4. **What should I do?** — cultural, biological and chemical management, plus supplies to buy.

Every photo is **stored in a crop gallery** organised by crop and condition. People can confirm or
correct labels, and the images export as a training dataset. The site is also the app's **store and
information website**: landing page, pricing plans, supplies shop with cart and checkout, About and
Contact pages. Payments go to Mobile Money **0200687324**.

Created by **Daniel Krause** — design, AI model, knowledge base, web app and store.

It uses the same `.tflite` model format as the FDCS Android (Kivy) app, so one trained model
serves both.

## Features

| Area | What it does |
|---|---|
| Capture | Live camera in the browser (front/back switch), file upload, drag-and-drop, phone camera via upload, bulk upload of up to 30 photos |
| Detection | Whole-image CNN classification **plus** overlapping-tile voting to box where the problem is; severity from affected area |
| Organism classification | Common + scientific name, organism type, kingdom, group, order/family, full taxonomy line, top-3 predictions |
| Safe to eat? | Verdict per condition, adjusted when the crop was sprayed recently; "Can't tell" when the model isn't confident |
| Crop gallery | All images stored and browsable by crop, condition and safety verdict; confirm/correct labels; export ZIP dataset |
| Advice | Identification tips, symptoms, lifecycle / spread, cultural / biological / chemical control |
| Quality checks | Warns about dark, over-exposed, blurry or low-resolution photos |
| History | Every scan with crop, location, notes, safety verdict; filter; delete |
| Pests & diseases library | 28 conditions + healthy, filter by pest / disease, search by name, organism or crop |
| Store | App plans (from GH₵250 / US$35 abroad), traps, tools, organic inputs, grain storage at Ghana market prices; prices in 20+ currencies at live exchange rates; checkout with payment to MoMo 0200687324 (direct transfer or Paystack) or cash on delivery |
| Owner page | `/admin`: orders, money received, payments to check, mark paid/delivered, contact messages, full crop gallery and dataset export |
| Security | CSRF tokens, strict Content-Security-Policy, secure cookies, rate limits, private scans per device, admin lockout, safe uploads |
| Website | Landing page, About (method, model, team, limitations), Contact form |
| API | `POST /api/predict` for the Android app, `GET /api/pests`, `GET /api/health` |
| Trained model | Included: 13 classes, **92.4 % test accuracy**, 7 MB TFLite; report in `models/report/` |
| Training | `training/train_quick.py` (laptop CPU, minutes) and `training/train.py` (full fine-tuning, GPU) |

## The trained model (included)

`models/pest_model.tflite` was trained from 8,585 PlantVillage leaf photos with
`training/train_quick.py`: ImageNet-pretrained MobileNetV2 extracts features from each photo plus 3
augmented copies, and a dense classifier (1024 units) is trained on them.

| Result on 1,288 test photos the model never saw | |
|---|---|
| Accuracy | **92.4 %** (macro F1 0.923) |
| Full app pipeline (tiles + fusion), 300 photos | 90.0 % |
| Speed | about 80 ms per photo on a 2-core CPU |
| Size | 7.2 MB (float16 TFLite) — also works in the Android app |

The confusion matrix, per-class precision/recall/F1 and training curves are in `models/report/`,
ready for your results chapter. `samples/` has six held-out photos to try in the app.

**Classes the model recognises (13):** healthy · maize grey leaf spot · maize leaf blight · maize
common rust · red spider mite · tomato bacterial spot (also pepper) · tomato early blight · tomato late
blight · tomato leaf mould · tomato mosaic virus · tomato septoria leaf spot · tomato target spot ·
tomato yellow leaf curl virus.

**Be honest about its limits:**
- PlantVillage photos are single leaves on plain backgrounds. Photos taken in a field (mixed leaves,
  soil, shadows) will score lower — collect them with the crop gallery and retrain.
- The library also profiles cassava, cocoa, insect pests and aflatoxin mould, marked *Not in model
  yet*. They need training photos (e.g. Kaggle *Cassava Leaf Disease*, IP102 insect pests, your own
  field photos). Put them in `dataset/<class_name>/` folders and rerun training.
- Photos the model isn't sure about (confidence below 0.30, calibrated on the test photos) are shown
  as *Not recognised* with a *Can't tell* safety verdict, instead of a guess.

## Quick start (Windows) — double-click

| File | What it does |
|---|---|
| **`start_public.bat`** | Starts the site **and** a free HTTPS link (`https://….trycloudflare.com`) that works on **any phone, anywhere**, with a QR code to scan. Live camera and "Install app" work. |
| `start_local.bat` | Starts the site for this computer and phones on the same Wi-Fi / hotspot only. |
| `allow_firewall.bat` | One-time fix if phones on the same Wi-Fi can't open the site (Windows blocks port 8000, especially on hotspots). Asks for admin permission. |

The first run creates a Python environment and installs packages (5–10 minutes), then asks for
your owner password (hidden as you type). Keep the window open — closing it stops the site.
The public link changes every time you start it; for a permanent address see *Putting it online*.

Command line instead: `python launcher.py public` (or `local`), or `python serve.py`.
`python app.py` now starts the same server on port **8000**.

### Why phones couldn't open it before
- **Different network:** `http://192.168…` / `http://172.20…` addresses only work for phones on the
  same Wi-Fi or hotspot as the laptop.
- **Windows Firewall** silently blocks other devices → `allow_firewall.bat`.
- **Port mix-up:** the old `app.py` used port 5000 while the tunnel pointed at 8000. Everything is on 8000 now.
- **No HTTPS:** phones only allow the live camera and app install on `https://` — the public link provides it.

## Install it like an app (PWA)

Open the public link on a phone:
- **Android (Chrome):** tap **Install** in the top bar (or ⋮ → *Install app*).
- **iPhone (Safari):** Share ⬆︎ → **Add to Home Screen**.

It gets its own icon, opens full-screen without the browser bar, loads fast on slow data (styles,
scripts and icons are cached) and shows an offline page when there's no signal; pages opened before
(like the pests & diseases library) still work offline. Crop checks need internet because the photo is
analysed on the server. Private pages (results, orders, history, owner area) are never stored on the phone.

## Retraining (and getting results for your report)

Arrange images one folder per class:

```
dataset/
  healthy/  maize_rust/  tomato_late_blight/  cassava_mosaic_disease/  ...
```

```bash
pip install -r requirements-train.txt
python training/train_quick.py --data dataset            # laptop CPU: ~10-15 minutes
python training/train.py --data dataset                  # full fine-tuning: best with a GPU (e.g. Google Colab)
```

`train_quick.py` extracts MobileNetV2 features once (plus `--aug-copies` augmented copies) and trains
the head on them; `--cache feats.npz` saves the features so you can retrain the head in about 2 minutes.
`train.py` fine-tunes the top MobileNetV2 layers too, which usually adds a few % accuracy.
Offline? Download the MobileNetV2 no-top weights once and pass `--weights path/to/file.h5`.

Both scripts use a stratified 70/15/15 split, class weighting, float16 TFLite export, and install
the model into `models/`. Reports land in `training/output/<timestamp>/`: `metrics.json`,
`classification_report.txt`, `confusion_matrix.png`, `training_curves.png`.

Folder names must match keys in `data/pests.json` to get organism and safe-to-eat information.
Evaluate any `.tflite` (including the Android app's) on labelled photos with timing:

```bash
python training/evaluate.py --data test_images --model models/pest_model.tflite
```

To use a model trained elsewhere, copy it to `models/pest_model.tflite`, make `models/labels.txt`
list the classes in the model's output order, and set `FDCS_NORMALIZE=mobilenet` if it expects
pixels scaled to −1..1 (`unit` for 0..1; the default `none` is for models from these scripts).

## Payments — all money goes to MoMo 0200687324

Settings live in `config.py` (or environment variables):

| Setting | Default | Meaning |
|---|---|---|
| `FDCS_MOMO_NUMBER` | `0200687324` | Wallet customers pay into |
| `FDCS_MOMO_NETWORK` | `Telecel Cash` | Shown to customers (020 numbers are Telecel) |
| `FDCS_MOMO_NAME` | *(empty)* | Name registered on the wallet — set it so customers can check they're paying the right person |
| `FDCS_ADMIN_PASSWORD` | *(empty = admin off)* | Password for the owner page at `/admin` |
| `FDCS_PAYSTACK_SECRET_KEY` | *(empty = off)* | Turns on automatic MoMo/card payments |

**Option 1 — Direct MoMo transfer (works now, no account needed).** After placing an order the
customer sees the amount, your number and a reference like `FDCS-12-A3F9C1`. They send the money
from their own phone (any network) and type in the transaction ID from their SMS. You then open
**/admin**, compare the ID and amount with the SMS on your phone, and click **Mark paid**. Only mark
an order paid after the money is in your wallet — anyone can type a fake transaction ID.

**Option 2 — Paystack (automatic).** Create a free business account at paystack.com (Ghana), add
**Mobile Money 0200687324** as the settlement account, copy your secret key, and start the app with:

```
set FDCS_PAYSTACK_SECRET_KEY=sk_live_xxxxxxxxx
set FDCS_ADMIN_PASSWORD=your-strong-password
python serve.py
```

Checkout then offers "Pay now with Mobile Money or card": the customer approves a prompt on their
phone, the app confirms the exact amount with Paystack's servers, and marks the order paid by itself.
Paystack deducts a small fee per transaction and pays out to your MoMo. In the Paystack dashboard,
set the webhook URL to `https://your-domain/payment/paystack/webhook` so payments are recorded even
if the customer closes the browser. Test first with an `sk_test_` key.

**Cash on delivery** is also offered for physical products. The admin page shows all orders,
amounts received, payments waiting to be checked, and contact messages.

Order pages use a secret link (`/order/<id>/<token>`) so customers can't see each other's details.

## Prices, countries and currencies

| Plan | In Ghana | Abroad (shown in the customer's currency) |
|---|---|---|
| FDCS Farmer | Free | Free |
| FDCS Pro Farmer | GH₵250 / month | US$35 / month |
| FDCS Cooperative | GH₵1,500 / month | US$150 / month |

Supplies are priced from Ghana market listings in September 2026 (Jiji Ghana, Supply Master) and
are delivered in Ghana only; items with no Ghana listing are marked *estimate* on the owner price
page, so check them with your supplier.

- The visitor's country is guessed from their browser language and can be changed with the
  "Shopping from" menu (store page and footer).
- Abroad, prices are converted with live exchange rates from open.er-api.com, refreshed every 12
  hours (saved in `instance/rates.json`; if offline, the last saved rates are used).
- Everything is **charged in Ghana cedis** to MoMo 0200687324 (or through Paystack), so a US$35
  plan is charged as its cedi equivalent — the customer sees both amounts.
- **Change prices any time** at **/admin → Edit prices**; changes apply immediately and are saved in
  the database. "Update rates now" fetches the latest exchange rates.

## Security

| Threat | Protection |
|---|---|
| Forged form submissions (CSRF) | Every form carries a secret per-session token; requests without it are rejected |
| Injected scripts (XSS) | Jinja auto-escaping + strict Content-Security-Policy (`script-src 'self'`, no inline scripts anywhere) |
| Clickjacking, MIME sniffing | `X-Frame-Options: DENY`, `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` |
| Session theft | HttpOnly + SameSite cookies, Secure + HSTS when `FDCS_HTTPS=1`, random 256-bit secret key saved in `instance/` |
| Password guessing | 5 admin login attempts per 15 min per IP, constant-time comparison, new session on login, 8-hour admin sessions |
| Snooping on other people | Scans, photos, locations and history are private to the device that made them; order pages need a secret link; full gallery and exports are owner-only |
| Malicious uploads | Size limit, file-type check, real image decoding, decompression-bomb guard, random file names, strict `/uploads/` name check |
| Flooding / spam | Rate limits on scans, bulk uploads, API, checkout, contact form |
| Fake payments | Paystack payments verified server-to-server (exact amount + currency); webhooks need a valid HMAC signature; direct MoMo payments are only marked paid by the owner |
| SQL injection | Parameterised queries only |
| Open redirects | Redirect targets restricted to this site |
| CSV formula injection | Exported spreadsheet cells can't start formulas |

Your part: choose a strong `FDCS_ADMIN_PASSWORD`, never share it or your Paystack secret key, serve
the site over HTTPS, and keep packages updated (`pip install -U -r requirements.txt`).
Optional: set `FDCS_API_KEY` so only your Android app can use `/api/predict`.

## Putting it online so anyone can use it

**Quick public link from your laptop (free, HTTPS):** double-click `start_public.bat`. It downloads
Cloudflare's `cloudflared` the first time, prints a `https://….trycloudflare.com` link with a QR code,
and keeps it running while the window stays open and the laptop stays awake.

**Permanent website — Render.com (recommended).**
1. Put the `fdcs_web` folder in a GitHub repository (the `.gitignore` keeps secrets, the database and
   uploads out).
2. On render.com: **New → Blueprint**, pick the repo. `render.yaml` sets everything up: HTTPS,
   production server, security settings, and a 1 GB persistent disk so the database and photos
   survive restarts.
3. In the Render dashboard, enter `FDCS_ADMIN_PASSWORD`, and optionally `FDCS_MOMO_NAME` and
   `FDCS_PAYSTACK_SECRET_KEY`. Render gives you an `https://fdcs.onrender.com`-style address; you
   can attach your own domain.
4. For automatic payments, set the Paystack webhook URL to `https://<your-address>/payment/paystack/webhook`.

The persistent disk requires Render's paid Starter plan. On the free plan, remove the `disk:` block
and the two `/var/data` paths from `render.yaml` — the site works, but stored photos and orders are
wiped whenever it restarts.

**Other hosts.** Any Linux host that runs Python: `pip install -r requirements.txt`, then
`gunicorn wsgi:app --workers 1 --threads 8`, behind HTTPS, with `FDCS_HTTPS=1` and
`FDCS_BEHIND_PROXY=1`. Keep **one worker** — rate limits and the model live in that process.

## Is it safe to eat? (how the verdict works)

Each condition in `data/pests.json` has an `edibility` entry (`safe`, `caution` or `unsafe`, with a
reason and advice). `fdcs/safety.py` then adjusts it:

- **Low confidence** (below `FDCS_MIN_CONFIDENCE`) → *Can't tell from this photo*. The app never
  guesses about food safety for something it doesn't recognise.
- **Sprayed in the last 2 weeks** (asked on the scan form) → *Safe* becomes *Eat with care*, with a
  reminder to wait for the product's pre-harvest interval. A verdict is never upgraded.
- **Label confirmed by a person** → the verdict follows the confirmed condition.

Examples: aflatoxin mould and cocoa black pod are *Not safe*; plant viruses (maize streak, cassava
mosaic) are *Safe* because they don't infect people; most insect damage is *Eat with care* (remove
damaged parts). The verdict covers only what is visible — it cannot detect pesticide residues or
invisible toxins, and the page says so.

## Crop gallery and dataset export

Every scanned or uploaded photo is kept in `uploads/` and recorded in SQLite. The **Crop gallery**
page groups them by crop (Tomato, Maize, Cassava…), filters by condition and safety verdict, and
accepts bulk uploads.

On any result page, **"Is this diagnosis right?"** lets a person confirm or correct the label. Then:

- **Export → Verified images only** downloads `condition_name/xxxx.jpg` folders plus `metadata.csv`
  (predicted vs confirmed label, confidence, crop, location) — feed it straight to `training/train.py`.
- **Export → All images** also includes confident model predictions, but never unconfirmed
  "not recognised" photos.

To show real photos in the store instead of icons, put them in `static/img/products/` and add
`"image": "file.jpg"` to the product in `data/products.json`.

## How detection works

The CNN is a classifier, so localisation is done by sliding-window classification
(`fdcs/detector.py`):

1. The whole image is classified.
2. The image is split into a 3×3 grid of 50%-overlapping tiles; each tile is classified.
3. Because tiles overlap, the image divides into a 4×4 grid of cells, each covered by up to four
   tiles. A cell's pest score is the average over the tiles covering it, so only cells that
   several tiles agree on pass the 0.55 threshold. Connected passing cells become one box, so
   separate infestations get separate boxes.
4. Final scores = 0.6 × whole image + 0.4 × strongest tile. If the whole image looks healthy but a
   tile is ≥ 0.75 sure of a pest, the pest is reported (small pests on large leaves).
5. Severity comes from the share of cells affected. Below 0.30 confidence (calibrated on test photos) the result is marked
   "low confidence — retake photo".

Tune with `FDCS_TILE_GRID`, `FDCS_TILE_THRESHOLD`, `FDCS_MIN_CONFIDENCE`.

## API (for the Android app or other clients)

```bash
curl -F "image=@leaf.jpg" -F "crop=Maize" http://127.0.0.1:8000/api/predict
```

Send `sprayed=yes|no|unsure` too for an accurate safety verdict. Returns `result` (label,
confidence, top_k, detections with boxes, severity, warnings), `category` (pest / disease /
healthy / unrecognised), `organism` (names, kingdom, group, taxonomy), `safe_to_eat` (status,
reason, advice), `management` and `annotated_url`.

## Camera notes

Browsers only allow live camera access on **HTTPS or localhost**. Testing on a phone over your
Wi-Fi (`http://192.168.x.x:8000`) the live camera won't open, but the **Upload** tab still opens
the phone camera. For a public demo, deploy behind HTTPS (Render, PythonAnywhere, Railway), or
use the Cloudflare tunnel described above.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Project structure

```
app.py               Flask app: scan, results, gallery, history, library, store, payments, admin, API
serve.py             Production server (waitress), port 8000, prints the phone address
launcher.py          One-command start: server + Cloudflare HTTPS link + QR code
start_public.bat, start_local.bat, allow_firewall.bat   Double-click helpers for Windows
static/sw.js, /manifest.webmanifest, static/icons/      Installable web app (PWA)
wsgi.py, Procfile, render.yaml   Linux hosting / one-click Render deploy
config.py            All settings (model path, normalisation, thresholds, team, contacts)
fdcs/model.py        TFLite/Keras classifier + demo fallback
fdcs/detector.py     Tiled detection, fusion, severity, quality checks, box drawing
fdcs/safety.py       Safe-to-eat verdict
fdcs/security.py     CSRF, security headers, rate limits, secret key, safe redirects
fdcs/payments.py     Paystack client (initialise, verify, webhook signatures)
fdcs/pricing.py      Country prices, live exchange rates
fdcs/db.py           SQLite: scans (+ verified labels), orders, messages
data/pests.json      Knowledge base: 29 entries, taxonomy, safe-to-eat data
data/products.json   Store catalogue
models/              pest_model.tflite (trained, 13 classes) + labels.txt + report/
samples/             Held-out test photos to try
training/            train_quick.py (CPU), train.py (GPU fine-tune), evaluate.py, report.py
templates/, static/  Pages, CSS, camera JavaScript
tests/               pytest suite
```

## Disclaimer

Pest management advice is general guidance. Always confirm with a MoFA extension officer, use only
products registered by the Ghana EPA (and COCOBOD/CRIG guidance for cocoa), follow label
instructions and wear protective equipment.
