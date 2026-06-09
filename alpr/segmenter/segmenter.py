"""
alpr/segmenter/segmenter.py
============================
API pública del segmentador de caràcters (Fase 2).

Pipeline per crop:
  deskew → binarize_adaptive → extract_contours → filter_geometric
  → remove_overlapping → is_plausible_plate → crops_and_resize

API pública:
  segment(roi_bgr)            -> list[np.ndarray]   [] si rebutjat
  segmenta_caixa(roi_bgr)     -> dict               totes les etapes
  save_chars(chars, ...)      -> None
"""

from pathlib import Path

import cv2
import numpy as np

from .deskew      import deskew
from .binarize    import binarize_adaptive
from .contours    import extract_contours, filter_geometric, remove_overlapping
from .validate    import is_plausible_plate
from .char_export import crops_and_resize

from alpr import config


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════

def segment(roi_bgr: np.ndarray, fmt: str | None = None) -> list[np.ndarray]:
    """
    Segmenta els caràcters d'un crop de matrícula.

    Retorna list[np.ndarray 28×28] (blanc sobre negre).
    Retorna [] si el crop no supera la validació geomètrica (no és matrícula).

    El pipeline retorna [] per a falsos positius del detector: és el mecanisme
    de rebuig de la Fase 2 (veure contractes a config.py / guia §4).
    """
    H, W = roi_bgr.shape[:2]

    aligned, _angle  = deskew(roi_bgr)
    thresh            = binarize_adaptive(aligned)
    all_bboxes        = extract_contours(thresh)
    filtered          = filter_geometric(all_bboxes)
    filtered          = remove_overlapping(filtered)
    accepted, _reason = is_plausible_plate(filtered, W, H)

    if not accepted:
        return []

    return crops_and_resize(thresh, filtered, fmt)


def segmenta_caixa(roi_bgr: np.ndarray, fmt: str | None = None) -> dict:
    """
    Com segment però retorna un dict complet amb totes les etapes.
    Útil per a diagnòstic i visualització.

    Claus del dict:
      aligned, angle, thresh,
      all_bboxes, filtered_bboxes,
      accepted, rejection_reason,
      chars  (list[np.ndarray 28×28] o [])
    """
    H, W = roi_bgr.shape[:2]

    aligned, angle   = deskew(roi_bgr)
    thresh            = binarize_adaptive(aligned)
    all_bboxes        = extract_contours(thresh)
    filtered          = filter_geometric(all_bboxes)
    filtered          = remove_overlapping(filtered)
    accepted, reason  = is_plausible_plate(filtered, W, H)
    chars             = crops_and_resize(thresh, filtered, fmt) if accepted else []

    return {
        "aligned":          aligned,
        "angle":            angle,
        "thresh":           thresh,
        "all_bboxes":       all_bboxes,
        "filtered_bboxes":  filtered,
        "accepted":         accepted,
        "rejection_reason": reason,
        "chars":            chars,
    }


def save_chars(
    chars: list[np.ndarray],
    roi_name: str,
    out_dir: Path | str,
    metadata: dict | None = None,
) -> None:
    """
    Guarda els caràcters 28×28 a disc.

    Nomenclatura: {roi_name}_char{i:02d}.png
    Si metadata conté 'gt' (string), s'afegeix al nom com a referència.

    Crea out_dir si no existeix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, char_img in enumerate(chars):
        fname = f"{roi_name}_char{i:02d}.png"
        cv2.imwrite(str(out_dir / fname), char_img)
