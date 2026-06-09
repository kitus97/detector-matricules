"""
alpr/ocr/dataset/augment.py
============================
Fase 3 — prep dades: augmentation sintètica i construcció del dataset.

Llegeix imatges base 64×64 (negre sobre blanc) i genera:
  data/synthetic/train/{clase}/*.png   (80 %)
  data/synthetic/val/{clase}/*.png     (20 %)

Cada imatge de sortida: 28×28, binària, caràcter BLANC sobre fons NEGRE
(mateixa convenció que el segmentador real).
"""

import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

# ── Probabilitats i rangs ─────────────────────────────────────────────────────
PROB_ROTATION  = 0.7
PROB_SHEAR     = 0.4
PROB_TRANSLATE = 0.5
PROB_SCALE     = 0.5
ROT_MAX_DEG    = 8.0
SHEAR_MAX      = 0.1
TRANSLATE_MAX  = 0.10
SCALE_RANGE    = (0.85, 1.15)

PROB_EROSION   = 0.3
PROB_DILATION  = 0.3
PROB_NOISE     = 0.4
NOISE_MAX_PCT  = 0.03

PROB_CROP_SIDE = 0.3
PROB_OFFSET    = 0.4
CROP_MAX_PX    = 3
OFFSET_MAX_PX  = 3


# ══════════════════════════════════════════════════════════════════════════════
# Grup A — Degradació geomètrica
# ══════════════════════════════════════════════════════════════════════════════

def apply_geometric(img: np.ndarray) -> np.ndarray:
    h, w  = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.eye(3, dtype=np.float64)

    if random.random() < PROB_SCALE:
        s  = random.uniform(*SCALE_RANGE)
        Ms = np.float64([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1]])
        M  = Ms @ M

    if random.random() < PROB_ROTATION:
        angle  = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
        rot_2x3 = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        Mr = np.vstack([rot_2x3, [0, 0, 1]])
        M  = Mr @ M

    if random.random() < PROB_SHEAR:
        shear = random.uniform(-SHEAR_MAX, SHEAR_MAX)
        Msh   = np.float64([[1, shear, -cy * shear], [0, 1, 0], [0, 0, 1]])
        M     = Msh @ M

    if random.random() < PROB_TRANSLATE:
        tx = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX) * w
        ty = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX) * h
        Mt = np.float64([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        M  = Mt @ M

    return cv2.warpAffine(
        img, M[:2], (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Grup B — Degradació de traç
# ══════════════════════════════════════════════════════════════════════════════

def apply_stroke(img: np.ndarray) -> np.ndarray:
    # Nota OpenCV: caràcter NEGRE sobre fons BLANC.
    # cv2.erode  → expandeix el negre → traç MÉS GRUIXUT
    # cv2.dilate → expandeix el blanc → traç MÉS PRIM
    img = img.copy()
    if random.random() < PROB_EROSION:
        ksize  = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        img    = cv2.dilate(img, kernel, iterations=1)
    elif random.random() < PROB_DILATION:
        ksize  = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        img    = cv2.erode(img, kernel, iterations=1)

    if random.random() < PROB_NOISE:
        n_px = max(1, int(random.uniform(0.01, NOISE_MAX_PCT) * img.size))
        rows = np.random.randint(0, img.shape[0], n_px // 2)
        cols = np.random.randint(0, img.shape[1], n_px // 2)
        img[rows, cols] = 255
        rows = np.random.randint(0, img.shape[0], n_px // 2)
        cols = np.random.randint(0, img.shape[1], n_px // 2)
        img[rows, cols] = 0
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Grup C — Degradació de contorn
# ══════════════════════════════════════════════════════════════════════════════

def apply_contour(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out  = img.copy()
    if random.random() < PROB_CROP_SIDE:
        px   = random.randint(1, CROP_MAX_PX)
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":    out[:px, :]     = 255
        elif side == "bottom": out[h - px:, :] = 255
        elif side == "left": out[:, :px]     = 255
        else:                out[:, w - px:] = 255

    if random.random() < PROB_OFFSET:
        dx = random.randint(-OFFSET_MAX_PX, OFFSET_MAX_PX)
        dy = random.randint(-OFFSET_MAX_PX, OFFSET_MAX_PX)
        Mt = np.float64([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, Mt, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Grup D — Binarització final (sempre s'aplica)
# ══════════════════════════════════════════════════════════════════════════════

def apply_binarize(img: np.ndarray) -> np.ndarray:
    """
    Mateixos paràmetres que el segmentador real (binarize.py).
    Inverteix a blanc-sobre-negre.
    """
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=config.ADAPT_BLOCK,
        C=config.ADAPT_C,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline d'augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment(img: np.ndarray) -> np.ndarray:
    """
    Orquestra A → B → C → D sobre una imatge 64×64 negre-sobre-blanc.
    Retorna 28×28 binari blanc-sobre-negre.
    """
    img = apply_geometric(img)
    img = apply_stroke(img)
    img = apply_contour(img)
    img = apply_binarize(img)
    return cv2.resize(img, (config.OUTPUT_SIZE, config.OUTPUT_SIZE),
                      interpolation=cv2.INTER_NEAREST)


# ══════════════════════════════════════════════════════════════════════════════
# Construcció del dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    samples_per_class: int = 500,
    val_split: float = 0.2,
) -> None:
    """
    Genera el dataset augmentat amb estructura train/val.
    Guarda class_to_idx.json a out_dir.
    """
    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No s'han trobat carpetes de classe a '{raw_dir}'")

    classes      = [d.name for d in class_dirs]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    n_val        = max(1, int(samples_per_class * val_split))
    n_train      = samples_per_class - n_val

    log.info(f"Classes: {len(classes)}  |  train={n_train}  val={n_val}")

    total_ok = 0
    for cls in classes:
        base_imgs = sorted((raw_dir / cls).glob("*.png"))
        if not base_imgs:
            log.warning(f"  SALT '{cls}': sense imatges base")
            continue

        (out_dir / "train" / cls).mkdir(parents=True, exist_ok=True)
        (out_dir / "val"   / cls).mkdir(parents=True, exist_ok=True)

        for split, n_samples in [("train", n_train), ("val", n_val)]:
            for i in range(n_samples):
                base_path = base_imgs[i % len(base_imgs)]
                base_img  = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
                if base_img is None:
                    continue
                aug_img  = augment(base_img)
                out_path = out_dir / split / cls / f"img_{i:05d}.png"
                cv2.imwrite(str(out_path), aug_img)
                total_ok += 1

        log.info(f"  OK  '{cls}'  ({len(base_imgs)} bases → {n_train}+{n_val})")

    idx_path = out_dir / "class_to_idx.json"
    idx_path.write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False))
    log.info(f"Imatges generades: {total_ok}  |  class_to_idx: {idx_path}")
