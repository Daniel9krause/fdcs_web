"""Train the FDCS pest classifier (MobileNetV2 transfer learning) and export TFLite.

Dataset layout - one folder per class (folder names become the labels):

    dataset/
        aphids/            img001.jpg ...
        fall_armyworm/     ...
        healthy/           ...
        ...

Usage (from the project root):

    python training/train.py --data dataset --epochs 15 --fine-tune-epochs 10

Outputs go to training/output/<timestamp>/ and the deployable model is copied to
models/pest_model.tflite and models/labels.txt so the web app picks it up.

Produces the numbers for your results chapter: test accuracy, per-class
precision/recall/F1, a confusion matrix image and training curves.
"""
import argparse
import json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Folder with one sub-folder per class")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15, help="Epochs with the base frozen")
    p.add_argument("--fine-tune-epochs", type=int, default=10, help="Epochs with top layers unfrozen (0 to skip)")
    p.add_argument("--fine-tune-layers", type=int, default=40, help="How many top MobileNetV2 layers to unfreeze")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--weights", default="imagenet", help="'imagenet' (recommended) or 'none'")
    p.add_argument("--quantize", choices=["none", "float16", "dynamic"], default="float16",
                   help="TFLite optimisation: float16 halves the size with ~no accuracy loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-install", action="store_true", help="Don't copy the model into models/")
    return p.parse_args()


def list_images(data_dir):
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 class folders in {data_dir}, found {classes}")
    return classes, np.array(paths), np.array(labels)


def make_dataset(paths, labels, img_size, batch, training, seed):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32), label

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.15, seed=seed),
        tf.keras.layers.RandomZoom(0.15, seed=seed),
        tf.keras.layers.RandomContrast(0.2, seed=seed),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
    ], name="augmentation")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augment(x, training=True), 0, 255), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def build_model(n_classes, img_size, weights):
    """Input is raw 0-255 RGB; MobileNetV2 scaling (-1..1) is inside the model."""
    inputs = tf.keras.Input((img_size, img_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(inputs)
    base = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                             weights=None if weights == "none" else weights)
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="pest_class")(x)
    return tf.keras.Model(inputs, outputs, name="fdcs_mobilenetv2"), base


def export_tflite(model, path, quantize):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]
    with open(path, "wb") as fh:
        fh.write(converter.convert())
    return os.path.getsize(path)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    out_dir = os.path.join(ROOT, "training", "output", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    classes, paths, labels = list_images(args.data)
    print(f"Found {len(paths)} images in {len(classes)} classes: {classes}")

    # stratified train / val / test split, saved for reproducibility
    tr_p, te_p, tr_y, te_y = train_test_split(paths, labels, test_size=args.test_split,
                                              stratify=labels, random_state=args.seed)
    tr_p, va_p, tr_y, va_y = train_test_split(tr_p, tr_y, test_size=args.val_split / (1 - args.test_split),
                                              stratify=tr_y, random_state=args.seed)
    with open(os.path.join(out_dir, "split.json"), "w") as fh:
        json.dump({"classes": classes, "train": tr_p.tolist(), "val": va_p.tolist(), "test": te_p.tolist()}, fh)
    counts = {c: {"train": int((tr_y == i).sum()), "val": int((va_y == i).sum()), "test": int((te_y == i).sum())}
              for i, c in enumerate(classes)}
    print(f"Split: train={len(tr_p)} val={len(va_p)} test={len(te_p)}")

    train_ds = make_dataset(tr_p, tr_y, args.img_size, args.batch, True, args.seed)
    val_ds = make_dataset(va_p, va_y, args.img_size, args.batch, False, args.seed)
    test_ds = make_dataset(te_p, te_y, args.img_size, args.batch, False, args.seed)

    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=tr_y)
    class_weight = dict(enumerate(weights))

    model, base = build_model(len(classes), args.img_size, args.weights)
    best_path = os.path.join(out_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv"), append=True),
    ]

    # ---- phase 1: train the new classifier head
    model.compile(tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight,
                   callbacks=callbacks)
    history = {k: list(v) for k, v in h1.history.items()}
    phase1_epochs = len(h1.history["loss"])

    # ---- phase 2: fine-tune the top of MobileNetV2
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(tf.keras.optimizers.Adam(args.lr / 100), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=phase1_epochs,
                       epochs=phase1_epochs + args.fine_tune_epochs, class_weight=class_weight,
                       callbacks=callbacks)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(v)
    train_time = time.time() - t0

    # ---- evaluation on the untouched test set
    if os.path.exists(best_path):
        model = tf.keras.models.load_model(best_path)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    # ---- export
    model.save(os.path.join(out_dir, "pest_model.keras"))
    tflite_path = os.path.join(out_dir, "pest_model.tflite")
    size = export_tflite(model, tflite_path, args.quantize)
    with open(os.path.join(out_dir, "labels.txt"), "w") as fh:
        fh.write("\n".join(classes) + "\n")

    extra = {"test_loss": float(test_loss), "keras_test_accuracy": float(test_acc),
             "train_time_sec": round(train_time, 1), "tflite_size_mb": round(size / 1e6, 2),
             "quantize": args.quantize, "img_size": args.img_size, "epochs_run": len(history["loss"]),
             "class_counts": counts, "weights": args.weights}
    metrics = save_reports(out_dir, classes, te_y, y_pred, history, extra)

    if not args.no_install:
        shutil.copy(tflite_path, os.path.join(ROOT, "models", "pest_model.tflite"))
        shutil.copy(os.path.join(out_dir, "labels.txt"), os.path.join(ROOT, "models", "labels.txt"))
        print("Installed model into models/ - restart the web app to use it.")

    print(f"\nTest accuracy: {metrics['accuracy']:.4f}  macro-F1: {metrics['macro_f1']:.4f}")
    print(f"TFLite model: {size / 1e6:.2f} MB   Reports: {out_dir}")


if __name__ == "__main__":
    main()
