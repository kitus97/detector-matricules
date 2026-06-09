"""
scripts/experiments/sweep_detector.py
======================================
E5 — Sensibilitat del detector morfològic als seus llindars de forma
(memòria §5.2). Mostra el compromís fonamental **recall ↔ candidats/imatge**:
afluixar un filtre puja el recall però genera més falsos positius (que la Fase 2
ha de filtrar), i estrènyer-lo fa el contrari.

Sweeps (un knob cada cop, la resta per defecte):
  · EXTENT_MIN        — quanta àrea del bbox ha d'omplir el blob.
  · ASPECT_RATIO_MAX  — relació d'aspecte màxima admesa.

S'avalua amb el mateix harness de recall que `compare_detectors.py`.

Ús
--
    python scripts/experiments/sweep_detector.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from scripts.experiments.detectors import morphologic
from scripts.experiments.compare_detectors import load_gt_images, evaluate


def _print_sweep(items, knob: str, values: list, iou_thresh: float) -> list[dict]:
    header = f"{knob:<16} {'Recall(centre)':>14} {'Recall(IoU)':>12} " \
             f"{'Cand med':>9} {'Cand mitj':>10} {'Cand màx':>9}"
    print(header); print("─" * len(header))
    original = getattr(config, knob)
    rows = []
    for v in values:
        setattr(config, knob, v)
        m = evaluate(morphologic, items, iou_thresh)
        print(f"{knob}={v:<8} {m['recall_center']:>13.1%} {m['recall_iou']:>11.1%} "
              f"{m['cand_median']:>9.0f} {m['cand_mean']:>10.1f} {m['cand_max']:>9d}")
        rows.append({"knob": knob, "valor": v, "recall_center": f"{m['recall_center']:.4f}",
                     "recall_iou": f"{m['recall_iou']:.4f}", "cand_mean": f"{m['cand_mean']:.2f}",
                     "cand_max": m["cand_max"]})
    setattr(config, knob, original)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=ARREL / "output" / "experiments")
    args = parser.parse_args()

    import csv
    items = load_gt_images(args.data_dir)
    if not items:
        print(f"ERROR: cap imatge amb GT a {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*72}\nE5 — Sweep del detector morfològic sobre {len(items)} imatges "
          f"(IoU≥{args.iou_thresh})\n{'='*72}")
    print(f"\nDefault: EXTENT_MIN={config.EXTENT_MIN}, ASPECT_RATIO_MAX={config.ASPECT_RATIO_MAX}\n")

    rows = []
    rows += _print_sweep(items, "EXTENT_MIN", [0.15, 0.25, 0.35, 0.45], args.iou_thresh)
    print()
    rows += _print_sweep(items, "ASPECT_RATIO_MAX", [5.0, 7.0, 9.0, 12.0], args.iou_thresh)

    out = args.out_dir / "E5_detector_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV desat a: {out}")


if __name__ == "__main__":
    main()
