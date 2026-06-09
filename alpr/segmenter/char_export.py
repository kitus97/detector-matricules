"""
alpr/segmenter/char_export.py
==============================
Fase 2, pas 5: retall i exportació dels caràcters a 28×28.

Format "vj" (default i recomanat):
  Retall tight de thresh + resize 28×28 INTER_AREA.
  Validat empíricament com a molt millor per a l'OCR que el llenç 64×64.
"""

import cv2
import numpy as np

from alpr import config


def preprocessa_caracter(
    thresh: np.ndarray,
    bbox: tuple[int, int, int, int],
    fmt: str | None = None,
) -> np.ndarray:
    """
    Retalla un caràcter de la imatge thresh i el redimensiona al format demanat.

    Paràmetres
    ----------
    thresh : imatge binària (caràcters blancs sobre negre)
    bbox   : (x, y, w, h) del caràcter
    fmt    : "vj" (default) o "regla-or" (llenç 64×64 centrat)

    Retorna np.ndarray 28×28 uint8 binari (blanc sobre negre).
    """
    if fmt is None:
        fmt = config.FORMAT_SORTIDA

    x, y, w, h = bbox
    crop = thresh[y : y + h, x : x + w]

    if fmt == "vj":
        # Tight crop + resize  (millor OCR)
        return cv2.resize(
            crop,
            (config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    if fmt == "regla-or":
        # Caràcter centrat en llenç 64×64 → resize 28×28
        canvas = np.zeros((config.CANVAS_SIZE, config.CANVAS_SIZE), dtype=np.uint8)
        scale  = (config.CANVAS_SIZE - 2 * config.MARGIN) / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        ox = (config.CANVAS_SIZE - nw) // 2
        oy = (config.CANVAS_SIZE - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
        return cv2.resize(
            canvas,
            (config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    raise ValueError(f"Format desconegut: '{fmt}'. Usa 'vj' o 'regla-or'.")


def crops_and_resize(
    thresh: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    fmt: str | None = None,
) -> list[np.ndarray]:
    """
    Aplica preprocessa_caracter a tots els bboxes.
    Retorna llista de np.ndarray 28×28.
    """
    return [preprocessa_caracter(thresh, bbox, fmt) for bbox in bboxes]
