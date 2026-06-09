"""
scripts/experiments/error_attribution.py
=========================================
E9 — Atribució d'errors end-to-end (memòria §6, discussió).

Per cada imatge amb GT, segueix la matrícula pel pipeline i determina ON es perd:

  · detector_miss   → cap caixa detectada conté el centre del GT (Fase 1).
  · segmenter_reject→ la caixa del GT existeix però el segmentador la rebutja (Fase 2).
  · ocr_error       → se segmenta però la lectura final no coincideix amb el GT (Fase 3).
  · perfecta        → la lectura coincideix exactament amb el GT.

És el "funnel" que explica quants plats es perden a cada fase i on invertir
esforç (típicament: l'OCR, pel domain gap).

Ús
--
    python scripts/experiments/error_attribution.py
    python scripts/experiments/error_attribution.py --csv output/experiments/E9.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from alpr.common.io import iter_images
from alpr.common.annotations import load_annotations
from alpr.common.geometry import iou
from alpr.detector.detector import detect_boxes
from alpr.segmenter.segmenter import segment
from alpr.ocr.infer import load_model, predict
from alpr.reader.reader import read_plate


def _contains(box, cx, cy) -> bool:
    x, y, w, h = box
    return x <= cx <= x + w and y <= cy <= y + h


def attribute(img, gt_box, gt_plate, model, idx_to_class, device) -> tuple[str, str]:
    """Retorna (categoria, lectura) per a una imatge."""
    gx, gy, gw, gh = gt_box
    cx, cy = gx + gw / 2.0, gy + gh / 2.0

    boxes = detect_boxes(img)
    containing = [b for b in boxes if _contains(b, cx, cy)]
    if not containing:
        return "detector_miss", ""

    # La caixa del GT: la que conté el centre i té més IoU amb el GT
    box = max(containing, key=lambda b: iou(b, gt_box))
    x, y, w, h = box
    chars_imgs = segment(img[y:y + h, x:x + w])
    if not chars_imgs:
        return "segmenter_reject", ""

    preds = [predict(c, model, idx_to_class, device) for c in chars_imgs]
    chars_v = [ch for ch, _ in preds if ch != "?"]
    plate = read_plate(chars_v)
    if plate == gt_plate:
        return "perfecta", plate
    return "ocr_error", plate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--csv", type=Path, default=ARREL / "output" / "experiments" / "E9_atribucio.csv")
    args = parser.parse_args()

    model, _c2i, idx_to_class, device = load_model(None)

    cats = Counter()
    rows: list[dict] = []
    for path, img in iter_images(args.data_dir):
        anns = load_annotations(path.with_suffix(".txt"))
        if not anns:
            continue
        a = anns[0]
        gt_box = (a["x"], a["y"], a["w"], a["h"])
        gt_plate = a["plate"]
        cat, lectura = attribute(img, gt_box, gt_plate, model, idx_to_class, device)
        cats[cat] += 1
        rows.append({"imatge": path.name, "gt": gt_plate, "categoria": cat, "lectura": lectura})

    total = sum(cats.values())
    print(f"\n{'='*60}\nE9 — Atribució d'errors sobre {total} imatges amb GT\n{'='*60}")
    print(f"\n{'Categoria':<20} {'Imatges':>8} {'%':>7}")
    print("─" * 36)
    # Ordre del funnel
    for cat in ("perfecta", "ocr_error", "segmenter_reject", "detector_miss"):
        n = cats.get(cat, 0)
        print(f"{cat:<20} {n:>8} {n/total:>6.1%}")
    print("─" * 36)
    print(f"{'TOTAL':<20} {total:>8}")

    # Quants plats arriben a cada fase (funnel acumulat)
    detected = total - cats.get("detector_miss", 0)
    segmented = detected - cats.get("segmenter_reject", 0)
    print(f"\nFunnel:  detectades {detected}/{total} ({detected/total:.0%})  →  "
          f"segmentades {segmented}/{total} ({segmented/total:.0%})  →  "
          f"perfectes {cats.get('perfecta',0)}/{total} ({cats.get('perfecta',0)/total:.0%})")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["imatge", "gt", "categoria", "lectura"])
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV desat a: {args.csv}")


if __name__ == "__main__":
    main()
