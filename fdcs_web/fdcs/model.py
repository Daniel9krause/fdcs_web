"""CNN pest classifier.

Loads the trained model (TFLite preferred, Keras supported) and returns class
probabilities for a PIL image. If no model file or no inference runtime is
available the classifier runs in clearly-flagged DEMO mode so the rest of the
web application can still be developed and demonstrated.
"""
from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _load_tflite_interpreter(model_path: str):
    """Try every TFLite runtime that might be installed, lightest first."""
    errors = []
    for module in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow"):
        try:
            mod = __import__(module, fromlist=["Interpreter"])
            # tf.lite is a lazily-loaded attribute, not an importable sub-module
            cls = mod.lite.Interpreter if module == "tensorflow" else mod.Interpreter
            interpreter = cls(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter, module
        except Exception as exc:  # ImportError or bad model
            errors.append(f"{module}: {exc}")
    raise RuntimeError("No TFLite runtime could load the model:\n  " + "\n  ".join(errors))


class PestClassifier:
    def __init__(self, model_path: str, labels_path: str, input_size: int = 224,
                 normalize: str = "none"):
        self.model_path = model_path
        self.input_size = input_size
        self.normalize = normalize
        self.labels = self._read_labels(labels_path)
        self.backend = "demo"
        self.demo_reason = ""
        self._interpreter = None
        self._keras_model = None
        self._load()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _read_labels(path: str) -> list[str]:
        with open(path, encoding="utf-8") as fh:
            labels = [line.strip() for line in fh if line.strip()]
        if not labels:
            raise ValueError(f"No labels found in {path}")
        return labels

    def _load(self):
        if not os.path.exists(self.model_path):
            self.demo_reason = f"Model file not found: {os.path.basename(self.model_path)}"
            log.warning("DEMO MODE - %s", self.demo_reason)
            return
        try:
            if self.model_path.endswith(".tflite"):
                self._interpreter, module = _load_tflite_interpreter(self.model_path)
                self._in = self._interpreter.get_input_details()[0]
                self._out = self._interpreter.get_output_details()[0]
                h, w = int(self._in["shape"][1]), int(self._in["shape"][2])
                self.input_size = h if h == w and h > 0 else self.input_size
                n_out = int(self._out["shape"][-1])
                self.backend = f"tflite ({module})"
            else:
                import tensorflow as tf  # noqa: WPS433
                self._keras_model = tf.keras.models.load_model(self.model_path, compile=False)
                shape = self._keras_model.input_shape
                if shape[1]:
                    self.input_size = int(shape[1])
                n_out = int(self._keras_model.output_shape[-1])
                self.backend = "keras"
            if n_out != len(self.labels):
                raise ValueError(f"Model has {n_out} outputs but labels.txt has "
                                 f"{len(self.labels)} labels - they must match")
            log.info("Loaded %s model with %d classes", self.backend, n_out)
        except Exception as exc:
            self._interpreter = self._keras_model = None
            self.backend = "demo"
            self.demo_reason = str(exc).splitlines()[0][:200]
            log.error("Could not load model, falling back to DEMO MODE: %s", exc)

    @property
    def is_demo(self) -> bool:
        return self.backend == "demo"

    # ------------------------------------------------------------- inference
    def _prepare(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.normalize == "mobilenet":
            arr = arr / 127.5 - 1.0
        elif self.normalize == "unit":
            arr = arr / 255.0
        return arr[np.newaxis, ...]

    @staticmethod
    def _softmax_if_needed(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float64)
        if logits.min() >= 0 and abs(logits.sum() - 1.0) < 1e-2:
            return logits
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        if self.is_demo:
            return self._demo_proba(image)
        x = self._prepare(image)
        if self._interpreter is not None:
            dtype = self._in["dtype"]
            if dtype in (np.uint8, np.int8):  # quantised model
                scale, zero = self._in["quantization"]
                scale = scale or 1.0
                x = np.clip(np.round(x / scale + zero), np.iinfo(dtype).min,
                            np.iinfo(dtype).max).astype(dtype)
            else:
                x = x.astype(dtype)
            self._interpreter.set_tensor(self._in["index"], x)
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._out["index"])[0]
            if self._out["dtype"] in (np.uint8, np.int8):
                scale, zero = self._out["quantization"]
                out = (out.astype(np.float32) - zero) * scale
        else:
            out = self._keras_model.predict(x, verbose=0)[0]
        return self._softmax_if_needed(np.asarray(out))

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        proba = self.predict_proba(image)
        order = np.argsort(proba)[::-1][:top_k]
        return [{"label": self.labels[i], "confidence": float(proba[i])} for i in order]

    # ------------------------------------------------------------------ demo
    def _demo_proba(self, image: Image.Image) -> np.ndarray:
        """Deterministic *simulated* output so the UI can be demonstrated.

        The same image always gives the same answer. Uniform green images lean
        towards 'healthy'. These numbers are NOT predictions.
        """
        small = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32)
        seed = int(hashlib.md5(small.astype(np.uint8).tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        proba = rng.dirichlet(np.full(len(self.labels), 0.12))
        r, g, b = small[..., 0].mean(), small[..., 1].mean(), small[..., 2].mean()
        greenness = max(0.0, (g - max(r, b)) / 255.0)
        texture = small.std() / 128.0
        if "healthy" in self.labels:
            idx = self.labels.index("healthy")
            proba[idx] += greenness * 4 * max(0.0, 1 - texture)
        return proba / proba.sum()

    def info(self) -> dict:
        return {"backend": self.backend, "demo": self.is_demo, "demo_reason": self.demo_reason,
                "classes": len(self.labels), "labels": self.labels,
                "input_size": self.input_size, "normalize": self.normalize,
                "model_file": os.path.basename(self.model_path)}
