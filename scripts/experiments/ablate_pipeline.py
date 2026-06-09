"""
scripts/experiments/ablate_pipeline.py
=======================================
Ablacions internes del segmentador (memòria §4.2):

  E3 — Normalització d'escala: sweep de NORM_H (D9 / ADR-002 / C3).
  E4 — Deskew: regressió de centroides vs cap vs minAreaRect (ADR-003 / C2).

Cada variant regenera els caràcters (contorns + vj) i s'avalua amb el mateix
arnès que `eval_real`. E3 sobreescriu `config.NORM_H`; E4 substitueix la funció
`deskew` que crida el segmentador (monkeypatch del nom de mòdul).

Ús
--
    python scripts/experiments/ablate_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from alpr.ocr.infer import load_model
from alpr.segmenter.deskew import deskew as deskew_centroides
import alpr.segmenter.segmenter as seg_mod
from scripts.experiments.ablate_segmenter import (
    generate_chars, run_eval, _HEADER, _row, _save_csv,
)


# ══════════════════════════════════════════════════════════════════════════════
# Variants de deskew per a E4
# ══════════════════════════════════════════════════════════════════════════════

def deskew_cap(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Sense deskew: retorna el crop tal qual."""
    return crop_bgr, 0.0


def deskew_minarearect(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Mètode rebutjat per l'ADR-003: minAreaRect global sobre el binari Otsu."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    points = np.column_stack(np.where(binary == 255))
    if len(points) < 5:
        return crop_bgr, 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(points[:, ::-1].astype(np.float32))
    angle_corr = angle + 90.0 if w < h else angle
    if abs(angle_corr) > config.DESKEW_ANGLE_MAX:
        return crop_bgr, 0.0
    H_img, W_img = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W_img / 2.0, H_img / 2.0), angle_corr, 1.0)
    aligned = cv2.warpAffine(crop_bgr, M, (W_img, H_img),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned, angle_corr


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--out-dir", type=Path, default=ARREL / "output" / "experiments")
    args = parser.parse_args()

    chars_dir = args.out_dir / "chars" / "_ablate_tmp"
    model, _c2i, idx_to_class, device = load_model(None)

    def eval_current(label: str):
        n = generate_chars(chars_dir, "contorns", "vj")
        m = run_eval(model, idx_to_class, device, chars_dir, args.conf)
        line, rc = _row(label, n, m)
        print(line)
        return rc

    # ── E3 — Sweep de NORM_H ───────────────────────────────────────────────────
    print(f"\n{'='*72}\nE3 — Normalització d'escala (sweep NORM_H · contorns/vj · "
          f"conf={args.conf})\n{'='*72}")
    print(_HEADER); print("─" * len(_HEADER))
    e3_rows = []
    orig_norm = config.NORM_H
    for nh in (32, 48, 64, 96, 128, 160):
        config.NORM_H = nh
        e3_rows.append(eval_current(f"NORM_H={nh}"))
    config.NORM_H = orig_norm
    _save_csv(e3_rows, args.out_dir / "E3_norm_h.csv")

    # ── E4 — Ablació del deskew ────────────────────────────────────────────────
    print(f"\n{'='*72}\nE4 — Deskew (NORM_H={config.NORM_H} · contorns/vj · "
          f"conf={args.conf})\n{'='*72}")
    print(_HEADER); print("─" * len(_HEADER))
    e4_rows = []
    for label, fn in (("centroides (actual)", deskew_centroides),
                      ("cap", deskew_cap),
                      ("minAreaRect (rebutjat)", deskew_minarearect)):
        seg_mod.deskew = fn
        e4_rows.append(eval_current(label))
    seg_mod.deskew = deskew_centroides
    _save_csv(e4_rows, args.out_dir / "E4_deskew.csv")

    print(f"\nCSV desats a: {args.out_dir}")


if __name__ == "__main__":
    main()
