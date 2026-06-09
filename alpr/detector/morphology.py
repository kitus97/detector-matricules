"""
alpr/detector/morphology.py
============================
Fase 1, pas 3: morfologia matemàtica + etiquetatge de components connexos.

Closing horitzontal: agrupa les vores verticals disperses en un blob compacte
que correspon a la zona de text de la matrícula.
Opening: elimina blobs massa fins (soroll residual).
"""

import cv2
import numpy as np

import config


def morphology_and_label(
    binary: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Aplica closing horitzontal → opening → connectedComponentsWithStats.

    Retorna (morph, num_labels, labels, stats).
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.CLOSE_KERNEL)
    closed       = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, config.OPEN_KERNEL)
    morph        = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        morph, connectivity=8
    )
    return morph, num_labels, labels, stats
