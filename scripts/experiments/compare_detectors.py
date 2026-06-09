"""
scripts/experiments/compare_detectors.py
=========================================
Experiment de Fase 1 — compara tècniques de detecció de matrícula amb el MATEIX
harness de recall sobre el ground truth de `data/raw/` (memòria §4.1.3 / §5.2).

Per a cada detector de `detectors.DETECTORS` mesura, sobre les mateixes imatges:
  · recall (centre)  → el centre del GT cau dins d'alguna caixa (mètrica laxa).
  · recall (IoU≥t)   → existeix una caixa amb IoU ≥ llindar (qualitat de localització).
  · candidats/imatge → mediana, mitjana, màxim (cost del recall).

IMPORTANT: recall i candidats/imatge s'han de llegir JUNTS — un detector amb més
recall però molts més candidats no és necessàriament millor (la Fase 2 ha de
filtrar tots aquests falsos positius).

Ús
--
    python scripts/experiments/compare_detectors.py
    python scripts/experiments/compare_detectors.py --iou-thresh 0.5 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from scripts.experiments.detectors import DETECTORS

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth i IoU
# ══════════════════════════════════════════════════════════════════════════════

def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_gt(txt_path: Path) -> list[tuple[int, int, int, int]]:
    """Llegeix les caixes GT d'un .txt (tabuladors o espais)."""
    boxes: list[tuple[int, int, int, int]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def load_gt_images(data_dir: Path) -> list[tuple[np.ndarray, tuple]]:
    """Carrega (imatge, gt_box) de totes les imatges amb anotació vàlida. Un cop."""
    items: list[tuple[np.ndarray, tuple]] = []
    for img_path in sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VALID_EXTS):
        gt_path = img_path.with_suffix(".txt")
        if not gt_path.exists():
            continue
        gts = parse_gt(gt_path)
        if not gts:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        items.append((img, gts[0]))   # gairebé totes les imatges tenen 1 matrícula
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Avaluació d'un detector
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(detect_fn, items, iou_thresh: float) -> dict:
    """Avalua un detector sobre items=[(img, gt_box)]. Retorna les mètriques."""
    n = len(items)
    hit_center = hit_iou = 0
    n_boxes: list[int] = []
    for img, gt in items:
        boxes = detect_fn(img)
        n_boxes.append(len(boxes))
        gx, gy, gw, gh = gt
        cx, cy = gx + gw / 2.0, gy + gh / 2.0
        if any(x <= cx <= x + w and y <= cy <= y + h for (x, y, w, h) in boxes):
            hit_center += 1
        if any(_iou(gt, b) >= iou_thresh for b in boxes):
            hit_iou += 1
    arr = np.array(n_boxes) if n_boxes else np.array([0])
    return {
        "n":             n,
        "recall_center": hit_center / n if n else 0.0,
        "recall_iou":    hit_iou / n if n else 0.0,
        "cand_median":   float(np.median(arr)),
        "cand_mean":     float(arr.mean()),
        "cand_max":      int(arr.max()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ARREL / "data" / "raw",
                        help="Directori amb imatges + .txt GT")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="Llindar d'IoU per al recall estricte (default: 0.5)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Desa la taula de resultats en aquest CSV")
    args = parser.parse_args()

    items = load_gt_images(args.data_dir)
    if not items:
        print(f"ERROR: cap imatge amb GT a {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nComparació de detectors sobre {len(items)} imatges amb GT "
          f"(IoU≥{args.iou_thresh})\n")
    header = f"{'Tècnica':<20} {'Recall(centre)':>14} {'Recall(IoU)':>12} " \
             f"{'Cand med':>9} {'Cand mitj':>10} {'Cand màx':>9}"
    print(header)
    print("─" * len(header))

    rows: list[dict] = []
    for name, fn in DETECTORS.items():
        m = evaluate(fn, items, args.iou_thresh)
        print(f"{name:<20} {m['recall_center']:>13.1%} {m['recall_iou']:>11.1%} "
              f"{m['cand_median']:>9.0f} {m['cand_mean']:>10.1f} {m['cand_max']:>9d}")
        rows.append({"detector": name, **{k: v for k, v in m.items() if k != "n"}})

    print()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Taula desada a: {args.csv}")


if __name__ == "__main__":
    main()
