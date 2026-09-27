"""Shared reporting: metrics.json, classification report, confusion matrix and curves."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)


def plot_confusion(cm, classes, path, normalize=True):
    data = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    size = max(6, len(classes) * 0.7)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(data, cmap="Greens", vmin=0, vmax=1 if normalize else None)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=[c.replace("_", " ") for c in classes], yticklabels=[c.replace("_", " ") for c in classes],
           xlabel="Predicted class", ylabel="True class",
           title="Normalised confusion matrix (test set)" if normalize else "Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = data[i, j]
            if val:
                ax.text(j, i, f"{val:.2f}" if normalize else int(val), ha="center", va="center", fontsize=8,
                        color="white" if val > (0.5 if normalize else data.max() / 2) else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(history.get(key, []), label="Training")
        ax.plot(history.get(f"val_{key}", []), label="Validation")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reports(out_dir, classes, y_true, y_pred, history=None, extra=None):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, digits=4, zero_division=0)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(), "classes": classes,
    }
    metrics.update(extra or {})
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as fh:
        fh.write(report)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix.png"))
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion_matrix_counts.png"), normalize=False)
    if history:
        plot_history(history, os.path.join(out_dir, "training_curves.png"))
    print(report)
    return metrics
