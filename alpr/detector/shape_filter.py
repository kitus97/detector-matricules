"""
alpr/detector/shape_filter.py
==============================
Fase 1, pas 4: filtre geomètric de components connexos.

Filosofia de màxim recall: els llindars són deliberadament laxos per no
perdre la matrícula real. És preferible tenir falsos positius que perdre
la matrícula, ja que el segmentador els descartarà a la fase 2.
"""

import numpy as np

import config


def filter_by_shape(
    num_labels: int,
    stats: np.ndarray,
    img_shape: tuple,
) -> list[tuple[int, int, int, int]]:
    """
    Filtra els components connexos per àrea relativa, aspect ratio i extent.

    Paràmetres
    ----------
    num_labels : resultat de connectedComponentsWithStats
    stats      : array de stats de cv2.connectedComponentsWithStats
    img_shape  : (H, W[, C]) de la imatge original

    Retorna
    -------
    list[(x, y, w, h)] en píxels de la imatge original.
    """
    H, W     = img_shape[:2]
    img_area = H * W
    boxes: list[tuple[int, int, int, int]] = []

    for lbl in range(1, num_labels):          # etiqueta 0 = fons
        x, y, w, h, area = stats[lbl]

        if w < config.MIN_WIDTH or h < config.MIN_HEIGHT:
            continue

        area_ratio = area / img_area
        if not (config.AREA_RATIO_MIN <= area_ratio <= config.AREA_RATIO_MAX):
            continue

        aspect = w / float(h)
        if not (config.ASPECT_RATIO_MIN <= aspect <= config.ASPECT_RATIO_MAX):
            continue

        extent = area / float(w * h)
        if extent < config.EXTENT_MIN:
            continue

        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes
