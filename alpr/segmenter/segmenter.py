"""
alpr/segmenter/segmenter.py
============================
API pública del segmentador de caràcters (Fase 2).

Pipeline per crop:
  normalitza_escala → deskew → binarize_adaptive → extract_contours → filter_geometric
  → select_dominant_row → remove_overlapping → is_plausible_plate → crops_and_resize

API pública:
  segment(roi_bgr)            -> list[np.ndarray]   [] si rebutjat
  segmenta_caixa(roi_bgr)     -> dict               totes les etapes
  save_chars(chars, ...)      -> None
"""

from pathlib import Path

import cv2
import numpy as np

from .normalize   import normalitza_escala
from .deskew      import deskew
from .binarize    import binarize_adaptive
from .contours    import (
    extract_contours, filter_geometric, select_dominant_row, remove_overlapping,
)
from .projection  import detecta_projeccio
from .validate    import is_plausible_plate
from .char_export import crops_and_resize

from alpr import config


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════

def segment(
    roi_bgr: np.ndarray,
    fmt: str | None = None,
    metode: str = "contorns",
) -> list[np.ndarray]:
    """
    Segmenta els caràcters d'un crop de matrícula.

    Retorna list[np.ndarray 28×28] (blanc sobre negre).
    Retorna [] si el crop no supera la validació geomètrica (no és matrícula).

    El pipeline retorna [] per a falsos positius del detector: és el mecanisme
    de rebuig de la Fase 2 (veure contractes a config.py / guia §4).

    `metode`: "contorns" (oficial, ADR-002) o "projeccio" (alternatiu, per comparar).
    """
    # Normalitza l'escala abans de detectar (D9/ADR-002): imprescindible perquè
    # blockSize=31 sigui coherent amb caràcters de plaques de 18–46 px. H i W es
    # prenen de la imatge normalitzada perquè els bboxes (i la cobertura
    # d'is_plausible_plate) quedin al mateix espai de coordenades.
    roi_bgr = normalitza_escala(roi_bgr)
    H, W = roi_bgr.shape[:2]

    aligned, _angle  = deskew(roi_bgr)
    thresh            = binarize_adaptive(aligned)
    all_bboxes        = (detecta_projeccio(thresh) if metode == "projeccio"
                         else extract_contours(thresh))
    filtered          = filter_geometric(all_bboxes, H)
    filtered          = select_dominant_row(filtered)
    filtered          = remove_overlapping(filtered)
    accepted, _reason = is_plausible_plate(filtered, W)

    if not accepted:
        return []

    # El format "regla-or" re-binaritza el glif en grisos (Grup D); el "vj" usa
    # el binari. Passem tots dos: el gray és l'aligned normalitzat i deskew-at.
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return crops_and_resize(thresh, gray, filtered, fmt)


def segmenta_caixa(
    roi_bgr: np.ndarray,
    fmt: str | None = None,
    metode: str = "contorns",
) -> dict:
    """
    Com segment però retorna un dict complet amb totes les etapes.
    Útil per a diagnòstic i visualització.

    Claus del dict:
      aligned, angle, thresh,
      all_bboxes, filtered_bboxes,
      accepted, rejection_reason,
      chars  (list[np.ndarray 28×28] o [])
    """
    # Normalització d'escala prèvia (D9/ADR-002), idèntica a segment().
    roi_bgr = normalitza_escala(roi_bgr)
    H, W = roi_bgr.shape[:2]

    aligned, angle   = deskew(roi_bgr)
    thresh            = binarize_adaptive(aligned)
    all_bboxes        = (detecta_projeccio(thresh) if metode == "projeccio"
                         else extract_contours(thresh))
    filtered          = filter_geometric(all_bboxes, H)
    filtered          = select_dominant_row(filtered)
    filtered          = remove_overlapping(filtered)
    accepted, reason  = is_plausible_plate(filtered, W)
    gray              = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    chars             = crops_and_resize(thresh, gray, filtered, fmt) if accepted else []

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
