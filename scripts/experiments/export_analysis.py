"""
scripts/experiments/export_analysis.py
=======================================
Exporta els resultats finals del pipeline en format **tidy** (una observació per
fila) per a anàlisi estadística posterior (boxplots, histogrames, etc.).

Genera dos CSV:

  analysis_plates.csv  — UNA fila per matrícula (ROI acceptada):
      stem, box, gt, pred, gt_length, n_segmented, n_after_filter, n_discarded,
      delta_len, edit_distance, char_acc_ed, plate_match, lengths_match,
      n_correct_pos, char_acc_pos, mean_conf, median_conf, min_conf, max_conf

  analysis_chars.csv   — UNA fila per caràcter segmentat:
      stem, box, char_idx, pred, confidence, filtered_out, gt, correct,
      group_gt, group_gt_length, group_edit_distance, group_plate_match

Idees d'anàlisi que habiliten:
  · boxplot de `confidence` separat per `correct` (correctes vs incorrectes).
  · boxplot d'`edit_distance` o `char_acc_ed` per `gt_length`.
  · boxplot de `mean_conf` per `plate_match`.
  · histograma de `delta_len` (sobre/sub-segmentació).

Per defecte regenera els caràcters des de `data/raw` amb el pipeline actual
(contorns + vj) per garantir que el CSV reflecteix el codi d'ara.

Ús
--
    python scripts/experiments/export_analysis.py
    python scripts/experiments/export_analysis.py --chars-dir data/chars   # no regenera
    python scripts/experiments/export_analysis.py --conf 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from alpr.ocr.infer import load_model
from alpr.ocr.eval_real import group_files, evaluate_group
from scripts.experiments.ablate_segmenter import generate_chars


def _conf_stats(retained: list[dict]) -> dict:
    """Estadístics de confiança sobre els caràcters retinguts d'una matrícula."""
    confs = [p["confidence"] for p in retained]
    if not confs:
        return {"mean_conf": "", "median_conf": "", "min_conf": "", "max_conf": ""}
    return {
        "mean_conf":   f"{float(np.mean(confs)):.4f}",
        "median_conf": f"{float(np.median(confs)):.4f}",
        "min_conf":    f"{float(np.min(confs)):.4f}",
        "max_conf":    f"{float(np.max(confs)):.4f}",
    }


def build_rows(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """A partir dels resultats per grup d'eval_real, construeix les files
    tidy per matrícula i per caràcter."""
    plate_rows: list[dict] = []
    char_rows: list[dict] = []

    for r in results:
        retained = [p for p in r["all_preds"] if not p["filtered_out"]]
        char_acc_ed = max(0.0, 1.0 - r["edit_distance"] / r["gt_length"]) if r["gt_length"] else 0.0
        n_correct_pos = sum(1 for p in r["positional_pairs"] if p["correct"])
        char_acc_pos = (n_correct_pos / len(r["positional_pairs"])
                        if r["positional_pairs"] else "")

        plate_rows.append({
            "stem":            r["stem"],
            "box":             r["box"],
            "gt":              r["gt"],
            "pred":            r["pred_str"],
            "gt_length":       r["gt_length"],
            "n_segmented":     r["n_segmented"],
            "n_after_filter":  r["n_after_filter"],
            "n_discarded":     r["n_discarded"],
            "delta_len":       r["n_after_filter"] - r["gt_length"],
            "edit_distance":   r["edit_distance"],
            "char_acc_ed":     f"{char_acc_ed:.4f}",
            "plate_match":     int(r["plate_match"]),
            "lengths_match":   int(r["lengths_match"]),
            "n_correct_pos":   n_correct_pos if r["positional_pairs"] else "",
            "char_acc_pos":    f"{char_acc_pos:.4f}" if char_acc_pos != "" else "",
            **_conf_stats(retained),
        })

        # Mapa GT per als retinguts quan les longituds coincideixen
        gt_map: dict[int, str] = {}
        if r["lengths_match"]:
            for pred_info, gt_char in zip(retained, r["gt"]):
                gt_map[id(pred_info)] = gt_char

        for p in r["all_preds"]:
            if r["lengths_match"] and not p["filtered_out"]:
                gt_char = gt_map.get(id(p), "?")
                correct = int(gt_char == p["pred"])
            else:
                gt_char, correct = "?", ""
            char_rows.append({
                "stem":                r["stem"],
                "box":                 r["box"],
                "char_idx":            p["char_idx"],
                "pred":                p["pred"],
                "confidence":          f"{p['confidence']:.4f}",
                "filtered_out":        int(p["filtered_out"]),
                "gt":                  gt_char,
                "correct":             correct,
                "group_gt":            r["gt"],
                "group_gt_length":     r["gt_length"],
                "group_edit_distance": r["edit_distance"],
                "group_plate_match":   int(r["plate_match"]),
            })

    return plate_rows, char_rows


def _save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows):>4} files → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                        help=f"Llindar de confiança (default: {config.CONF_THRESHOLD})")
    parser.add_argument("--chars-dir", type=Path, default=None,
                        help="Usa aquest dir de caràcters en lloc de regenerar")
    parser.add_argument("--out-dir", type=Path, default=ARREL / "output" / "experiments")
    args = parser.parse_args()

    if args.chars_dir is None:
        chars_dir = args.out_dir / "chars" / "_analysis"
        n_roi = generate_chars(chars_dir, "contorns", "vj")
        print(f"Caràcters regenerats (contorns/vj): {n_roi} ROIs")
    else:
        chars_dir = args.chars_dir
        print(f"Usant caràcters existents de: {chars_dir}")

    model, _c2i, idx_to_class, device = load_model(None)

    groups, _ = group_files(chars_dir, config.DATA_RAW_DIR)
    results = []
    for g in groups:
        r = evaluate_group(model, g, args.conf, idx_to_class, device, strict_length=False)
        if r is not None:
            results.append(r)

    plate_rows, char_rows = build_rows(results)

    print(f"\nExportant (conf={args.conf}):")
    _save(plate_rows, args.out_dir / "analysis_plates.csv")
    _save(char_rows, args.out_dir / "analysis_chars.csv")

    # Resum ràpid per orientar l'anàlisi
    n = len(plate_rows)
    perfect = sum(p["plate_match"] for p in plate_rows)
    confs_ok = [float(c["confidence"]) for c in char_rows if c["correct"] == 1]
    confs_err = [float(c["confidence"]) for c in char_rows if c["correct"] == 0]
    print(f"\nResum: {n} matrícules, {perfect} perfectes ({perfect/n:.0%})")
    if confs_ok and confs_err:
        print(f"  Confiança mediana — correctes: {np.median(confs_ok):.3f} · "
              f"incorrectes: {np.median(confs_err):.3f}")


if __name__ == "__main__":
    main()
