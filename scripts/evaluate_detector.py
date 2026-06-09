"""
evaluate_detector.py
=====================
Mesura el RECALL del detector morfològic de matrícules (Fase 1) contra el ground
truth de `data/raw/`. És l'script que genera els números de detecció de
`docs/situacio-actual.md` (recall ≈ 98% per centre, ≈ 80% per IoU≥0.5).

El detector NO calcula cap mètrica al seu notebook (`notebooks/01_morphologic_VJ.ipynb`):
només visualitza. Aquí en reimplementem les funcions **verbatim** (mateixos paràmetres)
i les avaluem sobre totes les imatges que tenen anotació `.txt`.

Dues definicions de recall, complementàries:
  · "centre"  → el centre del GT cau dins d'alguna caixa detectada (mètrica laxa;
                el detector és de màxim recall i retorna caixes generoses).
  · "IoU"     → existeix una caixa amb IoU ≥ --iou_thresh amb el GT (mètrica estricta,
                qualitat real de localització).

Ground truth (un .txt per imatge): filename ⇥ x ⇥ y ⇥ w ⇥ h ⇥ matricula

Ús
--
    python scripts/evaluate_detector.py
    python scripts/evaluate_detector.py --data_dir data/raw --iou_thresh 0.5 --show_failures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

# Arrel del projecte (aquest fitxer viu a <arrel>/scripts/)
ARREL = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ARREL / "data" / "raw"

# ─── Paràmetres del detector (verbatim de 01_morphologic_VJ.ipynb) ───────────
GAUSS_KSIZE      = (5, 5)
GAUSS_SIGMA      = 1.0
CLAHE_CLIP       = 2.0
CLAHE_TILE       = (8, 8)
CLOSE_KERNEL     = (15, 3)   # closing horitzontal: fusiona vores verticals en un blob
OPEN_KERNEL      = (5, 5)    # opening: elimina estructures fines espúries

AREA_RATIO_MIN   = 0.001     # 0.1 % de la imatge
AREA_RATIO_MAX   = 0.10      # 10 % de la imatge
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 9.0
EXTENT_MIN       = 0.25
MIN_WIDTH        = 40
MIN_HEIGHT       = 10

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("evaluate_detector")


# ══════════════════════════════════════════════════════════════════════════════
# Detector morfològic (verbatim del notebook)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Gris → suavitzat Gaussià → CLAHE. Retorna gris uint8 equalitzat."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSS_KSIZE, GAUSS_SIGMA)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(blurred)


def sobel_vertical_binary(gray: np.ndarray) -> np.ndarray:
    """Sobel vertical (vores verticals dels caràcters) + binarització Otsu."""
    sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel_x)
    _, binary = cv2.threshold(sobel_abs, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphology_and_label(binary: np.ndarray) -> tuple[int, np.ndarray]:
    """Closing horitzontal → opening → etiquetatge de components connexos (8-conn).
    Retorna (num_labels, stats)."""
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KERNEL)
    morph = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    return num_labels, stats


def filter_by_shape(num_labels: int, stats: np.ndarray,
                    img_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Conserva els candidats amb forma plausible (àrea, AR, extent, mida).
    Retorna bounding boxes (x, y, w, h). Filtres laxos: filosofia de màxim recall."""
    H, W = img_shape[:2]
    img_area = H * W
    boxes: list[tuple[int, int, int, int]] = []
    for lbl in range(1, num_labels):          # 0 és el fons
        x, y, w, h, area = stats[lbl]
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
        if not (AREA_RATIO_MIN <= area / img_area <= AREA_RATIO_MAX):
            continue
        if not (ASPECT_RATIO_MIN <= w / float(h) <= ASPECT_RATIO_MAX):
            continue
        if area / float(w * h) < EXTENT_MIN:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def detect(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Pipeline complet del detector → llista de bounding boxes candidates."""
    enhanced = preprocess(img_bgr)
    binary = sobel_vertical_binary(enhanced)
    num_labels, stats = morphology_and_label(binary)
    return filter_by_shape(num_labels, stats, img_bgr.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth i IoU
# ══════════════════════════════════════════════════════════════════════════════

def iou(box_a, box_b) -> float:
    """Intersection over Union entre dues caixes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_gt(txt_path: Path) -> list[tuple[int, int, int, int]]:
    """Llegeix les caixes GT d'un .txt (tabuladors o espais). Retorna [(x,y,w,h), ...]."""
    boxes: list[tuple[int, int, int, int]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        except ValueError:
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# Avaluació
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DATA_RAW_DIR,
                        help=f"Directori amb imatges + .txt GT (default: {DATA_RAW_DIR})")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="Llindar d'IoU per al recall estricte (default: 0.5)")
    parser.add_argument("--show_failures", action="store_true",
                        help="Llista les imatges on es perd la matrícula (recall per centre)")
    args = parser.parse_args()

    imatges = sorted(p for p in args.data_dir.iterdir()
                     if p.suffix.lower() in VALID_EXTS)
    if not imatges:
        log.error(f"Cap imatge a {args.data_dir}")
        return

    n_total = 0
    hit_center = 0
    hit_iou = 0
    n_boxes: list[int] = []
    fallades: list[str] = []

    for img_path in imatges:
        gt_path = img_path.with_suffix(".txt")
        if not gt_path.exists():
            continue
        gts = parse_gt(gt_path)
        if not gts:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning(f"No s'ha pogut llegir {img_path.name}")
            continue

        n_total += 1
        boxes = detect(img)
        n_boxes.append(len(boxes))

        gt = gts[0]                            # gairebé totes les imatges tenen 1 matrícula
        gx, gy, gw, gh = gt
        cx, cy = gx + gw / 2.0, gy + gh / 2.0

        encert_centre = any(x <= cx <= x + w and y <= cy <= y + h
                            for (x, y, w, h) in boxes)
        encert_iou = any(iou(gt, b) >= args.iou_thresh for b in boxes)

        if encert_centre:
            hit_center += 1
        else:
            fallades.append(img_path.name)
        if encert_iou:
            hit_iou += 1

    if n_total == 0:
        log.error("Cap imatge amb GT vàlid trobada.")
        return

    boxes_arr = np.array(n_boxes)
    log.info("─" * 60)
    log.info(f"DETECTOR — avaluació sobre {n_total} imatges amb ground truth")
    log.info("─" * 60)
    log.info(f"Recall (centre del GT dins d'alguna caixa): "
             f"{hit_center}/{n_total} = {hit_center / n_total:.1%}")
    log.info(f"Recall (IoU ≥ {args.iou_thresh}):              "
             f"{hit_iou}/{n_total} = {hit_iou / n_total:.1%}")
    log.info(f"Candidats per imatge: mediana={int(np.median(boxes_arr))}, "
             f"mitjana={boxes_arr.mean():.1f}, màx={int(boxes_arr.max())}")

    if args.show_failures and fallades:
        log.info(f"Imatges on es perd la matrícula ({len(fallades)}):")
        for nom in fallades:
            log.info(f"    {nom}")


if __name__ == "__main__":
    main()
