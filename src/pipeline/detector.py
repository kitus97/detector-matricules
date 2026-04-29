from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# =============================================================================
# PARÁMETROS (más estables)
# =============================================================================

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 6.5

MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.25

# kernels morfológicos
KERNEL_CLOSE = (17, 3)
KERNEL_DILATE = (5, 5)


# =============================================================================
# PIPELINE
# =============================================================================

def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    gray = clahe.apply(gray)

    return gray


def detect_edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    edges = cv2.Canny(blurred, lower, upper)
    return edges


def connect_regions(edges: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_DILATE)
    closed = cv2.dilate(closed, kernel2, iterations=1)

    return closed


# def find_candidates(binary: np.ndarray, image_shape) -> list[dict]:
#     contours, _ = cv2.findContours(
#         binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     h_img, w_img = image_shape[:2]
#     img_area = float(h_img * w_img)

#     candidates = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)

#         if h == 0:
#             continue

#         aspect_ratio = w / float(h)
#         area = w * h
#         area_ratio = area / img_area

#         if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
#             continue

#         if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
#             continue

#         candidates.append({
#             "x1": int(x),
#             "y1": int(y),
#             "x2": int(x + w),
#             "y2": int(y + h),
#             "area": int(area),
#             "aspect_ratio": float(aspect_ratio),
#         })

#     return candidates

def find_candidates(binary: np.ndarray, image_shape, gray: np.ndarray) -> list[dict]:
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(8000)

    regions, _ = mser.detectRegions(gray)

    h_img, w_img = image_shape[:2]

    boxes = []

    # --- convertir regiones a bounding boxes ---
    for r in regions:
        x, y, w, h = cv2.boundingRect(r)

        if h == 0 or w == 0:
            continue

        ar = w / float(h)

        # filtro tipo carácter
        if 0.2 < ar < 1.0 and 10 < h < 80:
            boxes.append((x, y, w, h))

    # =========================
    # AGRUPAR CAJAS (CLAVE)
    # =========================
    candidates = []

    for i in range(len(boxes)):
        x1, y1, w1, h1 = boxes[i]

        group = [boxes[i]]

        for j in range(len(boxes)):
            if i == j:
                continue

            x2, y2, w2, h2 = boxes[j]

            # cerca en vertical
            if abs(y1 - y2) < 15:
                # cerca en horizontal
                if abs(x1 - x2) < 100:
                    group.append(boxes[j])

        if len(group) < 4:
            continue

        # bounding box del grupo
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[0] + b[2] for b in group]
        hs = [b[1] + b[3] for b in group]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(ws)
        y_max = max(hs)

        w_box = x_max - x_min
        h_box = y_max - y_min

        if h_box == 0:
            continue

        ar = w_box / float(h_box)

        if 2.5 < ar < 7.0:
            candidates.append({
                "x1": int(x_min),
                "y1": int(y_min),
                "x2": int(x_max),
                "y2": int(y_max),
                "area": int(w_box * h_box),
                "aspect_ratio": float(ar),
            })

    return candidates


def draw_candidates(bgr: np.ndarray, candidates: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for c in candidates:
        cv2.rectangle(
            out,
            (c["x1"], c["y1"]),
            (c["x2"], c["y2"]),
            (0, 255, 0),
            2,
        )
    return out


# =============================================================================
# DEBUG
# =============================================================================

def _show_debug(title, bgr, gray, edges, morph, overlay):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(title)

    axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("CLAHE")

    axes[0, 2].imshow(edges, cmap="gray")
    axes[0, 2].set_title("Canny")

    axes[1, 0].imshow(morph, cmap="gray")
    axes[1, 0].set_title("Morfología")

    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Candidatos")

    axes[1, 2].axis("off")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# =============================================================================
# API
# =============================================================================

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(path: str | Path) -> Iterable[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in _IMG_EXT:
                yield f
    else:
        raise FileNotFoundError(f"Path no existent: {path}")


def _process_one(image_path: Path, verbose: bool) -> list[dict]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        if verbose:
            print(f"[WARN] No s'ha pogut llegir: {image_path}")
        return []

    gray = preprocess_image(bgr)
    edges = detect_edges(gray)
    morph = connect_regions(edges)
    # candidates = find_candidates(morph, bgr.shape)

    candidates = find_candidates(None, bgr.shape, gray)

    if verbose:
        print(f"[{image_path.name}] candidats: {len(candidates)}")
        overlay = draw_candidates(bgr, candidates)
        _show_debug(image_path.name, bgr, gray, edges, morph, overlay)

    return candidates


def generate_plate_candidates(
    image_path_or_dir: str | os.PathLike,
    verbose: bool = True,
) -> dict[str, list[dict]]:

    results = {}

    for img_path in _iter_images(image_path_or_dir):
        results[img_path.name] = _process_one(img_path, verbose)

    return results