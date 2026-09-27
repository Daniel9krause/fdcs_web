"""Fast CPU-friendly training (no GPU needed) - MobileNetV2 feature extraction.

Instead of pushing every image through the whole network on every epoch, this
script runs the frozen, ImageNet-pretrained MobileNetV2 once per image (plus a few
augmented copies), stores the 1280-number feature vector, and trains the
classification head on those vectors. The head is then attached to MobileNetV2
to make one normal model, exported to TFLite for the web app and Android app.

On a laptop CPU this trains in minutes instead of hours, and on PlantVillage-style
images it reaches similar accuracy to full fine-tuning.

    python training/train_quick.py --data dataset
    python training/train_quick.py --data dataset --weights mobilenet_v2_no_top.h5   # offline

Outputs: training/output/<timestamp>/ (metrics.json, classification_report.txt,
confusion_matrix.png, training_curves.png, pest_model.keras/.tflite, labels.txt),
and the model is installed into models/ unless --no-install.
"""
import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from report import save_reports  # noqa: E402
from train import export_tflite, list_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Folder with one sub-folder per class")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--aug-copies", type=int, default=2, help="Augmented copies of each training image")
    p.add_argument("--epochs", type=int, default=60, help="Max epochs for the head (early stopping)")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or path to MobileNetV2 no-top .h5")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--hidden", type=int, default=512, help="Units in the hidden layer of the head (0 = none)")
    p.add_argument("--cache", default="", help="Save/load extracted features here (.npz) to retrain the head quickly")
    p.add_argument("--quantize", choices=["none", "float16", "dynamic"], default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-install", action="store_true")
    return p.parse_args()


def image_ds(paths, img_size, batch, augment=None):
    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.cast(tf.image.resize(img, (img_size, img_size)), tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    if augment is not None:
        ds = ds.map(lambda x: tf.clip_by_value(augment(x, training=True), 0, 255), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    out_dir = os.path.join(ROOT, "training", "output", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    classes, paths, labels = list_images(args.data)
    print(f"Found {len(paths)} images in {len(classes)} classes")
    tr_p, te_p, tr_y, te_y = train_test_split(paths, labels, test_size=args.test_split, stratify=labels,
                                              random_state=args.seed)
    tr_p, va_p, tr_y, va_y = train_test_split(tr_p, tr_y, test_size=args.val_split / (1 - args.test_split),
                                              stratify=tr_y, random_state=args.seed)
    print(f"Split: train={len(tr_p)} val={len(va_p)} test={len(te_p)}")

    # --- frozen feature extractor: raw 0-255 RGB in, 1280 features out
    inputs = tf.keras.Input((args.img_size, args.img_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(inputs)
    base = tf.keras.applications.MobileNetV2(input_shape=(args.img_size, args.img_size, 3), include_top=False,
                                             weights=None if args.weights == "none" else args.weights,
                                             pooling="avg")
    base.trainable = False
    extractor = tf.keras.Model(inputs, base(x, training=False), name="extractor")

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=args.seed),
        tf.keras.layers.RandomRotation(0.2, seed=args.seed),
        tf.keras.layers.RandomZoom((-0.25, 0.1), seed=args.seed),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=args.seed),
        tf.keras.layers.RandomContrast(0.3, seed=args.seed),
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255), seed=args.seed),
    ])

    def features(p, aug=None, name=""):
        t = time.time()
        f = extractor.predict(image_ds(p, args.img_size, args.batch, aug), verbose=0)
        print(f"  features {name}: {f.shape} in {time.time() - t:.0f}s")
        return f

    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache)
        x_tr, y_tr, x_va, x_te = c["x_tr"], c["y_tr"], c["x_va"], c["x_te"]
        print(f"Loaded cached features from {args.cache}")
    else:
        print("Extracting features (runs MobileNetV2 once per image)…")
        x_tr = [features(tr_p, name="train")]
        y_tr = [tr_y]
        for i in range(args.aug_copies):
            x_tr.append(features(tr_p, augment, f"train aug {i + 1}"))
            y_tr.append(tr_y)
        x_tr, y_tr = np.concatenate(x_tr), np.concatenate(y_tr)
        x_va, x_te = features(va_p, name="val"), features(te_p, name="test")
        if args.cache:
            np.savez_compressed(args.cache, x_tr=x_tr, y_tr=y_tr, x_va=x_va, x_te=x_te)

    # --- classification head
    hidden = [tf.keras.layers.Dense(args.hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
              tf.keras.layers.Dropout(0.4)] if args.hidden else []
    head = tf.keras.Sequential([
        tf.keras.Input((x_tr.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        *hidden,
        tf.keras.layers.Dense(len(classes), activation="softmax",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="pest_class"),
    ], name="head")
    head.compile(tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)))
    hist = head.fit(x_tr, y_tr, validation_data=(x_va, va_y), epochs=args.epochs, batch_size=128,
                    class_weight=cw, verbose=2, callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)])

    # --- one deployable model: image -> MobileNetV2 -> head
    model = tf.keras.Model(inputs, head(base(x, training=False)), name="fdcs_mobilenetv2")
    y_pred = head.predict(x_te, verbose=0).argmax(axis=1)

    # sanity check: the combined model must agree with the feature pipeline
    check = model.predict(image_ds(te_p[:64], args.img_size, 64), verbose=0).argmax(axis=1)
    agree = float((check == y_pred[:64]).mean())
    print(f"End-to-end check on 64 test images: {agree:.0%} agreement")

    model.save(os.path.join(out_dir, "pest_model.keras"))
    tflite_path = os.path.join(out_dir, "pest_model.tflite")
    size = export_tflite(model, tflite_path, args.quantize)
    with open(os.path.join(out_dir, "labels.txt"), "w") as fh:
        fh.write("\n".join(classes) + "\n")

    counts = {c: {"train": int((tr_y == i).sum()), "val": int((va_y == i).sum()), "test": int((te_y == i).sum())}
              for i, c in enumerate(classes)}
    metrics = save_reports(out_dir, classes, te_y, y_pred, hist.history, {
        "method": "MobileNetV2 (ImageNet, frozen) feature extraction + dense head",
        "hidden_units": args.hidden,
        "aug_copies": args.aug_copies, "img_size": args.img_size, "epochs_run": len(hist.history["loss"]),
        "train_time_sec": round(time.time() - t0, 1), "tflite_size_mb": round(size / 1e6, 2),
        "quantize": args.quantize, "end_to_end_agreement": agree, "class_counts": counts})

    if not args.no_install:
        shutil.copy(tflite_path, os.path.join(ROOT, "models", "pest_model.tflite"))
        shutil.copy(os.path.join(out_dir, "labels.txt"), os.path.join(ROOT, "models", "labels.txt"))
        print("Installed model into models/ - restart the web app to use it.")
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}  macro-F1: {metrics['macro_f1']:.4f}  "
          f"({(time.time() - t0) / 60:.1f} min)\nReports: {out_dir}")


if __name__ == "__main__":
    main()
