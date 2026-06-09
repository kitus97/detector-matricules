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
) -> list[tuple[int, int, int, int]]:
    """
    Filtre d'alçada mediana (±15 %) i aspect ratio [AR_MIN, AR_MAX].

    AR mínim 0.05 per no perdre la 'I' (AR real ≈ 0.05–0.12).
    Retorna la llista filtrada ordenada per x.
    """
    heights = [h for (_, _, _, h) in bboxes if h > 10]
    if not heights:
        return []
    h_med  = float(np.median(heights))
    result = [
        (x, y, w, h) for (x, y, w, h) in bboxes
        if (h_med * config.H_CHAR_MIN_REL < h < h_med * config.H_CHAR_MAX_REL)
        and (config.AR_MIN < w / float(h) < config.AR_MAX)
    ]
    result.sort(key=lambda b: b[0])
    return result


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
