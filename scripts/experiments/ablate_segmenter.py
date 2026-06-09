"""
scripts/experiments/ablate_segmenter.py
========================================
Ablacions de la Fase 2 (segmentador) mesurades amb el MATEIX arnès que
`alpr.ocr.eval_real` (memòria §4.2 / §5.3 / §5.4).

Per cada variant: regenera els caràcters amb el pipeline real
(detector → segment amb la variant) a un directori temporal, classifica'ls amb
la CNN i compara amb el GT via distància d'edició. Així totes les variants es
comparen amb la mateixa metodologia (evita el biaix d'avaluacions barrejades que
adverteix `docs/segmenter_audit.md §1.5`).

Experiments:
  E1 — Mètode de segmentació:   contorns  vs  projecció        (§4.2.3)
  E2 — Format de sortida 28×28: vj        vs  regla-or         (§4.2.6 / ADR-005)
  E6 — Llindar de confiança:    sweep sobre el filtre post-OCR  (§5.4)

Ús
--
    python scripts/experiments/ablate_segmenter.py
    python scripts/experiments/ablate_segmenter.py --conf 0.5 --out-dir output/experiments
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from alpr.detector.detector import detect_boxes
from alpr.segmenter.segmenter import segment, save_chars
from alpr.common.io import iter_images
from alpr.ocr.infer import load_model
from alpr.ocr.eval_real import group_files, evaluate_group, compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Generació de caràcters i avaluació
# ══════════════════════════════════════════════════════════════════════════════

def generate_chars(out_dir: Path, metode: str, fmt: str) -> int:
    """Regenera els caràcters del dataset amb una variant del segmentador.
    Neteja out_dir abans. Retorna el nombre de ROIs acceptades."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n_roi = 0
    for path, img in iter_images(config.DATA_RAW_DIR):
        for i, (x, y, w, h) in enumerate(detect_boxes(img)):
            chars = segment(img[y:y + h, x:x + w], fmt=fmt, metode=metode)
            if chars:
                save_chars(chars, f"{path.stem}_box{i}", out_dir)
                n_roi += 1
    return n_roi


def run_eval(model, idx_to_class, device, chars_dir: Path, conf: float) -> dict:
    """Avalua els caràcters d'un directori amb un llindar de confiança donat."""
    groups, _ = group_files(chars_dir, config.DATA_RAW_DIR)
    results = []
    for g in groups:
        r = evaluate_group(model, g, conf, idx_to_class, device, strict_length=False)
        if r is not None:
            results.append(r)
    return compute_metrics(results)


# ══════════════════════════════════════════════════════════════════════════════
# Impressió de taules
# ══════════════════════════════════════════════════════════════════════════════

_HEADER = (f"{'Variant':<22} {'Grups':>6} {'CharAcc(ED)':>12} "
           f"{'PlateAcc':>9} {'CharPos':>8} {'GruposPos':>9}")


def _row(label: str, n_roi: int | None, m: dict) -> tuple[str, dict]:
    pos = f"{m['char_acc_pos']:.2%}" if m.get("char_acc_pos") is not None else "—"
    line = (f"{label:<22} {m['n_groups']:>6} {m['char_acc_ed']:>11.2%} "
            f"{m['plate_acc_ed']:>9.2%} {pos:>8} {m['n_pos_groups']:>9}")
    rowcsv = {
        "variant": label, "rois_acceptades": n_roi if n_roi is not None else "",
        "grups": m["n_groups"], "char_acc_ed": f"{m['char_acc_ed']:.4f}",
        "plate_acc_ed": f"{m['plate_acc_ed']:.4f}",
        "char_acc_pos": f"{m['char_acc_pos']:.4f}" if m.get("char_acc_pos") is not None else "",
        "n_plate_match": m["n_plate_match"],
    }
    return line, rowcsv


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                        help=f"Llindar de confiança per a E1/E2 (default: {config.CONF_THRESHOLD})")
    parser.add_argument("--out-dir", type=Path, default=ARREL / "output" / "experiments",
                        help="On desar els CSV i els chars temporals")
    args = parser.parse_args()

    chars_root = args.out_dir / "chars"
    model, _c2i, idx_to_class, device = load_model(None)

    # ── Genera els 3 jocs de caràcters necessaris (cada un un cop) ──────────────
    print("Generant caràcters per variant (detector → segment)…")
    sets = {
        ("contorns", "vj"):       chars_root / "contorns_vj",
        ("projeccio", "vj"):      chars_root / "projeccio_vj",
        ("contorns", "regla-or"): chars_root / "contorns_reglaor",
    }
    n_roi = {}
    for (metode, fmt), d in sets.items():
        n_roi[(metode, fmt)] = generate_chars(d, metode, fmt)
        print(f"  {metode:9s} · {fmt:8s} → {n_roi[(metode, fmt)]} ROIs acceptades")

    # ── E1 — Mètode: contorns vs projecció (fmt=vj) ────────────────────────────
    print(f"\n{'='*70}\nE1 — Mètode de segmentació (format vj, conf={args.conf})\n{'='*70}")
    print(_HEADER); print("─" * len(_HEADER))
    e1_rows = []
    for label, key in [("contorns", ("contorns", "vj")), ("projecció", ("projeccio", "vj"))]:
        m = run_eval(model, idx_to_class, device, sets[key], args.conf)
        line, rc = _row(label, n_roi[key], m); print(line); e1_rows.append(rc)
    _save_csv(e1_rows, args.out_dir / "E1_metode.csv")

    # ── E2 — Format de sortida: vj vs regla-or (metode=contorns) ───────────────
    print(f"\n{'='*70}\nE2 — Format de sortida 28×28 (mètode contorns, conf={args.conf})\n{'='*70}")
    print(_HEADER); print("─" * len(_HEADER))
    e2_rows = []
    for label, key in [("vj", ("contorns", "vj")), ("regla-or", ("contorns", "regla-or"))]:
        m = run_eval(model, idx_to_class, device, sets[key], args.conf)
        line, rc = _row(label, n_roi[key], m); print(line); e2_rows.append(rc)
    _save_csv(e2_rows, args.out_dir / "E2_format.csv")

    # ── E6 — Sweep del llindar de confiança (contorns/vj) ──────────────────────
    print(f"\n{'='*70}\nE6 — Llindar de confiança post-OCR (contorns/vj)\n{'='*70}")
    print(_HEADER); print("─" * len(_HEADER))
    e6_rows = []
    for conf in (0.0, 0.3, 0.5, 0.7, 0.9):
        m = run_eval(model, idx_to_class, device, sets[("contorns", "vj")], conf)
        line, rc = _row(f"conf={conf:.1f}", None, m); print(line); e6_rows.append(rc)
    _save_csv(e6_rows, args.out_dir / "E6_confianca.csv")

    print(f"\nCSV desats a: {args.out_dir}")


if __name__ == "__main__":
    main()
