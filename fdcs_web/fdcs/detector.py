"""Pest detection = classification of the whole image + localisation.

The CNN is an image *classifier*, so to show *where* on the crop the pest is we
run it over a grid of overlapping tiles (sliding-window detection). Tiles whose
top class is a pest above a threshold become bounding boxes; overlapping boxes
of the same class are merged. Whole-image and tile evidence are fused for the
final diagnosis.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
except ImportError:  # OpenCV is optional; only used for the blur check
    cv2 = None

PALETTE = ["#e63946", "#f4a261", "#9b5de5", "#00bbf9", "#f15bb5", "#fee440",
           "#00f5d4", "#ff6d00", "#8338ec", "#3a86ff", "#fb5607", "#ff006e"]


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in original-image pixels


@dataclass
class AnalysisResult:
    label: str
    confidence: float
    top_k: list[dict]
    detections: list[Detection] = field(default_factory=list)
    affected_ratio: float = 0.0
    severity: str = "none"
    warnings: list[str] = field(default_factory=list)
    is_demo: bool = False
    uncertain: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label, "confidence": round(self.confidence, 4),
            "top_k": [{"label": t["label"], "confidence": round(t["confidence"], 4)} for t in self.top_k],
            "detections": [{"label": d.label, "confidence": round(d.confidence, 4), "box": list(d.box)}
                           for d in self.detections],
            "affected_ratio": round(self.affected_ratio, 3), "severity": self.severity,
            "warnings": self.warnings, "is_demo": self.is_demo, "uncertain": self.uncertain,
        }


def load_image(data: bytes, max_side: int = 1600) -> Image.Image:
    """Open image bytes, fix phone camera rotation (EXIF) and cap the size."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def quality_warnings(img: Image.Image) -> list[str]:
    warnings = []
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    brightness = gray.mean()
    if brightness < 45:
        warnings.append("Image is very dark - try scanning in daylight.")
    elif brightness > 225:
        warnings.append("Image is over-exposed - avoid direct glare on the leaf.")
    if cv2 is not None:
        sharpness = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        if sharpness < 40:
            warnings.append("Image looks blurry - hold the camera steady and tap to focus.")
    if min(img.size) < 150:
        warnings.append("Image resolution is low - move closer or use a better photo.")
    return warnings


def _tiles(width: int, height: int, grid: int):
    """Overlapping tiles: each tile is 2/(grid+1) of the side, stride half a tile."""
    tw, th = int(width * 2 / (grid + 1)), int(height * 2 / (grid + 1))
    sx, sy = tw // 2, th // 2
    for row in range(grid):
        for col in range(grid):
            x1, y1 = col * sx, row * sy
            yield (x1, y1, min(x1 + tw, width), min(y1 + th, height))


def _vote_boxes(tile_probs: np.ndarray, cls: int, width: int, height: int, grid: int,
                threshold: float) -> tuple[list[tuple], list[float], float]:
    """Localise class `cls` by voting over overlapping tiles.

    Tiles overlap by half, so the image divides into (grid+1) x (grid+1) cells and
    each cell is covered by up to four tiles. A cell's score is the mean
    probability of `cls` over the tiles covering it; cells above `threshold`
    are joined into connected regions and each region becomes one box.
    Returns (boxes, box_confidences, affected_cell_ratio).
    """
    n = grid + 1
    tw, th = int(width * 2 / n), int(height * 2 / n)
    sx, sy = tw // 2, th // 2
    score = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            covering = [tile_probs[tr, tc, cls] for tr in (r - 1, r) for tc in (c - 1, c)
                        if 0 <= tr < grid and 0 <= tc < grid]
            score[r, c] = np.mean(covering)
    mask = score >= threshold
    seen = np.zeros_like(mask)
    boxes, confs = [], []
    for r in range(n):
        for c in range(n):
            if not mask[r, c] or seen[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:  # flood fill 4-connected cells
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < n and 0 <= nc < n and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows, cols = [x[0] for x in cells], [x[1] for x in cells]
            x1, y1 = min(cols) * sx, min(rows) * sy
            x2 = width if max(cols) == n - 1 else (max(cols) + 1) * sx
            y2 = height if max(rows) == n - 1 else (max(rows) + 1) * sy
            boxes.append((x1, y1, x2, y2))
            confs.append(float(max(score[x] for x in cells)))
    return boxes, confs, float(mask.mean())


def analyze(img: Image.Image, classifier, healthy_label: str = "healthy", grid: int = 3,
            tile_threshold: float = 0.55, min_confidence: float = 0.40) -> AnalysisResult:
    labels = classifier.labels
    full = classifier.predict_proba(img)

    # --- sliding-window classification over overlapping tiles
    n_tiles = grid * grid if grid > 1 else 0
    tile_probs = np.zeros((max(grid, 1), max(grid, 1), len(labels)))
    if n_tiles:
        for i, box in enumerate(_tiles(*img.size, grid)):
            tile_probs[i // grid, i % grid] = classifier.predict_proba(img.crop(box))
    tile_max = tile_probs.reshape(-1, len(labels)).max(axis=0) if n_tiles else np.zeros_like(full)

    # --- fuse whole-image and strongest-tile evidence
    fused = 0.6 * full + 0.4 * tile_max if n_tiles else full
    fused = fused / fused.sum()
    order = np.argsort(fused)[::-1]
    top_k = [{"label": labels[i], "confidence": float(fused[i])} for i in order[:3]]
    label, confidence = top_k[0]["label"], top_k[0]["confidence"]

    # If the whole image looks healthy but a tile is very sure of a pest, report the pest
    # (small pests on a large leaf are easily outvoted at whole-image scale).
    if label == healthy_label and n_tiles:
        flat = tile_probs.reshape(-1, len(labels)).copy()
        if healthy_label in labels:
            flat[:, labels.index(healthy_label)] = 0
        t, c = np.unravel_index(np.argmax(flat), flat.shape)
        if flat[t, c] >= 0.75:
            label, confidence = labels[c], float(flat[t, c])

    detections, affected = [], 0.0
    if label != healthy_label and n_tiles:
        boxes, confs, affected = _vote_boxes(tile_probs, labels.index(label), *img.size, grid, tile_threshold)
        detections = [Detection(label, cf, b) for b, cf in zip(boxes, confs)]

    if label == healthy_label:
        severity = "none"
    elif affected >= 0.5:
        severity = "high"
    elif affected >= 0.2 or confidence >= 0.8:
        severity = "moderate"
    else:
        severity = "low"

    return AnalysisResult(label=label, confidence=confidence, top_k=top_k, detections=detections,
                          affected_ratio=affected, severity=severity,
                          warnings=quality_warnings(img), is_demo=classifier.is_demo,
                          uncertain=confidence < min_confidence)


def annotate(img: Image.Image, result: AnalysisResult, names: dict[str, str]) -> Image.Image:
    """Draw detection boxes and labels on a copy of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, min(out.size) // 150)
    font_size = max(14, min(out.size) // 30)
    font = None
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    for i, det in enumerate(result.detections):
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle(det.box, outline=color, width=width)
        text = names.get(det.label, det.label)
        tb = draw.textbbox((det.box[0], det.box[1]), text, font=font)
        pad = 4
        y = det.box[1] - (tb[3] - tb[1]) - 2 * pad
        y = det.box[1] if y < 0 else y
        draw.rectangle((det.box[0], y, det.box[0] + (tb[2] - tb[0]) + 2 * pad,
                        y + (tb[3] - tb[1]) + 2 * pad), fill=color)
        draw.text((det.box[0] + pad, y + pad - (tb[1] - det.box[1])), text, fill="white", font=font)
    return out
