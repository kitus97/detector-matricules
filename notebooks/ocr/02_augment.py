"""
02_augment.py
=============
Pas 4 del pla d'entrenament OCR.

Llegeix les imatges base de data/synthetic/raw/{classe}/*.png (64×64 px,
escala de grisos, caràcter NEGRE sobre fons BLANC) i genera un dataset
augmentat estructurat per a entrenament:

    data/synthetic/
    ├── train/{classe}/*.png   (80%)
    └── val/{classe}/*.png     (20%)

Cada imatge de sortida: 28×28 px, escala de grisos, binària (0/255),
caràcter BLANC sobre fons NEGRE (convenció del segmentador real).

Ús
--
    python 02_augment.py
    python 02_augment.py --samples_per_class 500 --val_split 0.2 --seed 42
    python 02_augment.py --samples_per_class 20 --preview
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np


# ─── Constants: probabilitats i rangs d'augmentation ─────────────────────────

# Grup A — Degradació geomètrica
PROB_ROTATION  = 0.7
PROB_SHEAR     = 0.4
PROB_TRANSLATE = 0.5
PROB_SCALE     = 0.5
ROT_MAX_DEG    = 8.0       # ±8°
SHEAR_MAX      = 0.1       # ±0.1
TRANSLATE_MAX  = 0.10      # ±10% de la mida del llenç
SCALE_RANGE    = (0.85, 1.15)

# Grup B — Degradació de traç
PROB_EROSION   = 0.3       # erosió de tinta → traç més prim
PROB_DILATION  = 0.3       # dilatació de tinta → traç més gruixut
PROB_NOISE     = 0.4
NOISE_MAX_PCT  = 0.03      # fins a 3% de píxels afectats per soroll

# Grup C — Degradació de contorn
PROB_CROP_SIDE = 0.3
PROB_OFFSET    = 0.4
CROP_MAX_PX    = 3         # 0–3 px tallats d'un costat
OFFSET_MAX_PX  = 3         # ±3 px descentrat del caràcter

# Mides d'imatge
INPUT_SIZE  = 64           # mida de les imatges base
OUTPUT_SIZE = 28           # mida de sortida (igual que el segmentador)

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Grup A — Degradació geomètrica
# ══════════════════════════════════════════════════════════════════════════════

def apply_geometric(img: np.ndarray) -> np.ndarray:
    """
    Combina rotació, shear, desplaçament i escala en una sola matriu afí.

    borderValue=255: les zones buides s'omplen amb BLANC (fons), perquè en
    aquesta fase el caràcter és negre sobre blanc; no volem omplir de tinta.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = np.eye(3, dtype=np.float64)

    if random.random() < PROB_SCALE:
        s = random.uniform(*SCALE_RANGE)
        Ms = np.float64([[s, 0, cx * (1 - s)],
                         [0, s, cy * (1 - s)],
                         [0, 0, 1]])
        M = Ms @ M

    if random.random() < PROB_ROTATION:
        angle = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
        rot_2x3 = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        Mr = np.vstack([rot_2x3, [0, 0, 1]])
        M = Mr @ M

    if random.random() < PROB_SHEAR:
        shear = random.uniform(-SHEAR_MAX, SHEAR_MAX)
        # Shear centrat en cy per minimitzar el desplaçament vertical
        Msh = np.float64([[1, shear, -cy * shear],
                          [0, 1,     0],
                          [0, 0,     1]])
        M = Msh @ M

    if random.random() < PROB_TRANSLATE:
        tx = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX) * w
        ty = random.uniform(-TRANSLATE_MAX, TRANSLATE_MAX) * h
        Mt = np.float64([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]])
        M = Mt @ M

    out = cv2.warpAffine(img, M[:2], (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=255)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Grup B — Degradació de traç
# ══════════════════════════════════════════════════════════════════════════════

def apply_stroke(img: np.ndarray) -> np.ndarray:
    """
    CONVENCIÓ OPENCV (important):
      Treballem en l'espai "caràcter negre (0) sobre fons blanc (255)".
      cv2.erode  → calcula el mínim → el negre s'expandeix → traç MÉS GRUIXUT.
      cv2.dilate → calcula el màxim → el blanc s'expandeix → traç MÉS PRIM.
      Per tant:
        · "erosió de tinta" (traç prim)   → cv2.dilate
        · "dilatació de tinta" (traç gruixut) → cv2.erode
    """
    img = img.copy()

    # Erosió i dilatació són mútuament excloents en la mateixa imatge
    r_e = random.random()
    r_d = random.random()

    if r_e < PROB_EROSION:
        # Traç més prim: dilata el blanc (màxim del veïnat)
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    elif r_d < PROB_DILATION:
        # Traç més gruixut: eroda el blanc, expandeix el negre (mínim del veïnat)
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

    if random.random() < PROB_NOISE:
        noise_pct = random.uniform(0.01, NOISE_MAX_PCT)
        n_px = max(1, int(noise_pct * img.size))
        # Sal: píxels blancs aleatoris
        rows = np.random.randint(0, img.shape[0], n_px // 2)
        cols = np.random.randint(0, img.shape[1], n_px // 2)
        img[rows, cols] = 255
        # Pebre: píxels negres aleatoris
        rows = np.random.randint(0, img.shape[0], n_px // 2)
        cols = np.random.randint(0, img.shape[1], n_px // 2)
        img[rows, cols] = 0

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Grup C — Degradació de contorn
# ══════════════════════════════════════════════════════════════════════════════

def apply_contour(img: np.ndarray) -> np.ndarray:
    """
    Imita el tall de bounding box del segmentador i el descentrat del caràcter.
    Les zones descobertes s'omplen amb blanc (fons).
    """
    h, w = img.shape[:2]
    out = img.copy()

    if random.random() < PROB_CROP_SIDE:
        px = random.randint(1, CROP_MAX_PX)
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            out[:px, :] = 255
        elif side == "bottom":
            out[h - px:, :] = 255
        elif side == "left":
            out[:, :px] = 255
        else:
            out[:, w - px:] = 255

    if random.random() < PROB_OFFSET:
        dx = random.randint(-OFFSET_MAX_PX, OFFSET_MAX_PX)
        dy = random.randint(-OFFSET_MAX_PX, OFFSET_MAX_PX)
        Mt = np.float64([[1, 0, dx],
                         [0, 1, dy]])
        out = cv2.warpAffine(out, Mt, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Grup D — Binarització final (sempre s'aplica)
# ══════════════════════════════════════════════════════════════════════════════

def apply_binarize(img: np.ndarray) -> np.ndarray:
    """
    Pas crític: aplica EXACTAMENT els mateixos paràmetres que el segmentador real.

    THRESH_BINARY_INV: píxels per sota del llindar → 255, per sobre → 0.
    Com que el caràcter és negre (per sota del llindar blanc del fons),
    queda invertit: caràcter BLANC sobre fons NEGRE.
    Això garanteix que la textura de les vores sigui idèntica a la de les
    imatges reals que el model veurà durant la inferència.
    """
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline d'augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment(img: np.ndarray) -> np.ndarray:
    """
    Orquestra els quatre grups en ordre estricte:
      A (geometria) → B (traç) → C (contorn) → D (binarització)

    Entrada : 64×64, grisos, caràcter NEGRE sobre fons BLANC.
    Sortida : 28×28, binària (0/255), caràcter BLANC sobre fons NEGRE.

    Tot el processament es fa a 64×64; la reducció a 28×28 és l'últim pas
    per evitar que les transformacions geomètriques tallin el caràcter.
    """
    img = apply_geometric(img)
    img = apply_stroke(img)
    img = apply_contour(img)
    img = apply_binarize(img)
    img = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE),
                     interpolation=cv2.INTER_NEAREST)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Construcció del dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(raw_dir: Path, out_dir: Path,
                  samples_per_class: int, val_split: float) -> None:
    """
    Recorre les classes, genera les mostres augmentades, fa el split
    train/val i guarda les imatges i el fitxer class_to_idx.json.

    Si una classe té menys imatges base que samples_per_class, les bases
    es reutilitzen cíclicament amb augmentations diverses.
    """
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No s'han trobat carpetes de classe a '{raw_dir}'")

    classes = [d.name for d in class_dirs]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    n_val   = max(1, int(samples_per_class * val_split))
    n_train = samples_per_class - n_val

    log.info(f"Classes trobades   : {len(classes)}")
    log.info(f"Mostres per classe : {samples_per_class}  (train={n_train}, val={n_val})")
    log.info(f"Divisió validació  : {val_split:.0%}")
    log.info(f"Directori sortida  : {out_dir.resolve()}")
    log.info("-" * 50)

    total_ok = 0
    for cls in classes:
        base_imgs = sorted((raw_dir / cls).glob("*.png"))
        if not base_imgs:
            log.warning(f"  SALT  '{cls}': no hi ha imatges base")
            continue

        (out_dir / "train" / cls).mkdir(parents=True, exist_ok=True)
        (out_dir / "val"   / cls).mkdir(parents=True, exist_ok=True)

        for split, n_samples in [("train", n_train), ("val", n_val)]:
            for i in range(n_samples):
                # Distribució cíclica: si hi ha menys bases que mostres,
                # se'n generen múltiples variants de cada base
                base_path = base_imgs[i % len(base_imgs)]
                base_img = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
                if base_img is None:
                    log.warning(f"  ERROR llegint {base_path.name}")
                    continue

                aug_img = augment(base_img)
                out_path = out_dir / split / cls / f"img_{i:05d}.png"
                cv2.imwrite(str(out_path), aug_img)
                total_ok += 1

        log.info(f"  OK  '{cls}'  ({len(base_imgs)} bases → {n_train} train + {n_val} val)")

    idx_path = out_dir / "class_to_idx.json"
    idx_path.write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False))

    log.info("-" * 50)
    log.info(f"Imatges generades  : {total_ok}")
    log.info(f"class_to_idx       : {idx_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Preview visual (opcional, --preview)
# ══════════════════════════════════════════════════════════════════════════════

def preview(raw_dir: Path, n_chars: int = 8, n_variants: int = 5) -> None:
    """
    Graella de verificació: original (redimensionat) | N variants augmentades.
    Cada fila és una classe; la primera columna és la imatge base de referència.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no disponible. Salta el preview.")
        return

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    sample_classes = class_dirs[:n_chars]
    n_cols = 1 + n_variants

    fig, axes = plt.subplots(len(sample_classes), n_cols,
                              figsize=(n_cols * 1.5, len(sample_classes) * 1.5))

    for row, cls_dir in enumerate(sample_classes):
        base_imgs = sorted(cls_dir.glob("*.png"))
        if not base_imgs:
            continue
        base_img = cv2.imread(str(base_imgs[0]), cv2.IMREAD_GRAYSCALE)

        ax_row = axes[row] if len(sample_classes) > 1 else axes

        # Primera columna: original redimensionat per comparació visual
        orig_small = cv2.resize(base_img, (OUTPUT_SIZE, OUTPUT_SIZE),
                                interpolation=cv2.INTER_AREA)
        ax_row[0].imshow(orig_small, cmap="gray", vmin=0, vmax=255)
        if row == 0:
            ax_row[0].set_title("original", fontsize=8)
        ax_row[0].set_ylabel(cls_dir.name, fontsize=9, rotation=0, labelpad=14)
        ax_row[0].axis("off")

        # Columnes restants: variants augmentades
        for col in range(n_variants):
            aug_img = augment(base_img.copy())
            ax_row[col + 1].imshow(aug_img, cmap="gray", vmin=0, vmax=255)
            if row == 0:
                ax_row[col + 1].set_title(f"aug {col + 1}", fontsize=8)
            ax_row[col + 1].axis("off")

    plt.suptitle("Preview augmentation  (original | variants)", fontsize=11)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el dataset augmentat per a l'entrenament OCR."
    )
    parser.add_argument(
        "--raw_dir", default="data/synthetic/raw",
        help="Directori amb les imatges base (default: data/synthetic/raw)",
    )
    parser.add_argument(
        "--out_dir", default="data/synthetic",
        help="Directori de sortida (default: data/synthetic)",
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=500,
        help="Nombre total de mostres per classe (default: 500)",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Proporció per a validació (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Llavor aleatòria per a reproductibilitat (default: 42)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostra la graella visual d'augmentations i surt (no genera dataset)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not 0.0 < args.val_split < 1.0:
        raise ValueError(f"--val_split ha de ser entre 0 i 1, rebut: {args.val_split}")

    if args.preview:
        preview(raw_dir)
    else:
        build_dataset(raw_dir, out_dir, args.samples_per_class, args.val_split)


if __name__ == "__main__":
    main()
