"""Evaluate a trained .tflite / .keras model on a folder of labelled test images.

Uses exactly the same inference code as the web app, so the numbers match what
users experience. Also measures average inference time per image.

    python training/evaluate.py --data test_images --model models/pest_model.tflite
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fdcs.model import PestClassifier  # noqa: E402
from report import save_reports  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with one sub-folder per class")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "pest_model.tflite"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "models", "labels.txt"))
    ap.add_argument("--normalize", default="none", choices=["none", "mobilenet", "unit"])
    args = ap.parse_args()

    clf = PestClassifier(args.model, args.labels, normalize=args.normalize)
    if clf.is_demo:
        sys.exit(f"Could not load model: {clf.demo_reason}")
    classes = clf.labels
    y_true, y_pred, times = [], [], []
    for folder in sorted(os.listdir(args.data)):
        if folder not in classes:
            print(f"Skipping folder '{folder}' (not in labels.txt)")
            continue
        for f in sorted(os.listdir(os.path.join(args.data, folder))):
            if not f.lower().endswith(IMG_EXT):
                continue
            img = Image.open(os.path.join(args.data, folder, f)).convert("RGB")
            t0 = time.perf_counter()
            proba = clf.predict_proba(img)
            times.append(time.perf_counter() - t0)
            y_true.append(classes.index(folder))
            y_pred.append(int(np.argmax(proba)))
    if not y_true:
        sys.exit("No test images found.")
    out_dir = os.path.join(ROOT, "training", "output", "eval-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    m = save_reports(out_dir, classes, np.array(y_true), np.array(y_pred),
                     extra={"model": os.path.basename(args.model), "backend": clf.backend,
                            "n_images": len(y_true), "mean_inference_ms": round(1000 * float(np.mean(times)), 2)})
    print(f"Accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"{m['mean_inference_ms']} ms/image on {len(y_true)} images\nReports: {out_dir}")


if __name__ == "__main__":
    main()
