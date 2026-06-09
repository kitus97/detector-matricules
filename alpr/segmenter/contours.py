"""
alpr/segmenter/contours.py
===========================
Fase 2, pas 2–3: extracció de contorns, filtre geomètric i deduplicació IoU.
"""

import numpy as np
import cv2

from alpr import config
from alpr.common.geometry import iou


def extract_contours(thresh: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Extreu bboxes amb RETR_EXTERNAL.

    RETR_EXTERNAL ignora els forats interiors dels caràcters (O, 0, D, A, B…),
    evitant que el forat es compti com un caràcter separat.

    Retorna list[(x, y, w, h)].
    """
    cnts, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cv2.boundingRect(c) for c in cnts]


def filter_geometric(
    bboxes: list[tuple[int, int, int, int]],
    h_placa: int,
    strict: bool = True,
) -> list[tuple[int, int, int, int]]:
    """
    Filtre geomètric groller RELATIU A L'ALÇADA DE LA PLACA (`h_placa`), mai px
    absoluts. Elimina blobs massa petits/grans o amb aspect ratio implausible.

    En mode tolerant (`strict=False`, passada 1 per al deskew) eixampla els
    llindars; en mode estricte (`strict=True`, passada 2) els ajusta. Mateixa
    lògica que `filtra_geometric` del segmentador de referència.

    - alçada ∈ [H_CHAR_MIN_REL, H_CHAR_MAX_REL]·h_placa (×0.7/×1.2 si tolerant)
    - aspect ratio ∈ [AR_MIN, AR_MAX] (AR_MIN=0.05 preserva la 'I'; ×1.3 si tolerant)
    - àrea ≥ AREA_MIN_REL·h_placa²  (elimina speckle)

    Retorna la llista filtrada (sense ordenar; remove_overlapping ja ordena per x).
    """
    h_min    = config.H_CHAR_MIN_REL * (1.0 if strict else 0.7) * h_placa
    h_max    = config.H_CHAR_MAX_REL * (1.0 if strict else 1.2) * h_placa
    ar_max   = config.AR_MAX * (1.0 if strict else 1.3)
    area_min = config.AREA_MIN_REL * (h_placa ** 2)

    out: list[tuple[int, int, int, int]] = []
    for (x, y, w, h) in bboxes:
        if h <= 0 or w <= 0:
            continue
        if not (h_min <= h <= h_max):
            continue
        if not (config.AR_MIN <= w / float(h) <= ar_max):
            continue
        if (w * h) < area_min:
            continue
        out.append((x, y, w, h))
    return out


def select_dominant_row(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    De tots els candidats grollers, conserva els que tenen alçada propera a la
    MEDIANA (la fila dominant de caràcters), robust a alguns blobs de soroll.

    Banda relativa a la mediana: [1−H_MED_TOL, 1+H_MED_TOL]·mediana. Equival a
    `selecciona_fila_per_mediana` del segmentador de referència.
    """
    if not bboxes:
        return []
    med_h  = float(np.median([h for (_, _, _, h) in bboxes]))
    lo, hi = (1 - config.H_MED_TOL) * med_h, (1 + config.H_MED_TOL) * med_h
    return [b for b in bboxes if lo <= b[3] <= hi]


def remove_overlapping(
    bboxes: list[tuple[int, int, int, int]],
    iou_thresh: float | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Deduplicació per IoU: conserva el bbox de més àrea (contorn exterior).

    Ordena per àrea descendent perquè el contorn exterior sempre entri primer.
    Retorna la llista retinguda ordenada per x.
    """
    if iou_thresh is None:
        iou_thresh = config.IOU_OVERLAP

    if len(bboxes) <= 1:
        return list(bboxes)

    kept: list[tuple] = []
    for bbox in sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True):
        if not any(iou(bbox, k) > iou_thresh for k in kept):
            kept.append(bbox)
    kept.sort(key=lambda b: b[0])
    return kept
