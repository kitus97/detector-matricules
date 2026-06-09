"""
alpr/segmenter/binarize.py
===========================
Fase 2, pas 1: binarització adaptativa del crop alineat.

Usa EXACTAMENT els mateixos paràmetres que el dataset sintètic (02_augment.py)
per garantir que la textura de les vores sigui idèntica a la de les imatges
reals que el model OCR veurà durant la inferència.
"""

import cv2
import numpy as np

from alpr import config


def binarize_adaptive(aligned_bgr: np.ndarray) -> np.ndarray:
    """
    AdaptiveThreshold (MEAN_C, BINARY_INV) + MORPH_CLOSE vertical.

    El CLOSE consolida traços fins ('I') sense fusionar caràcters adjacents
    gràcies al kernel vertical (1×3).

    Retorna imatge binària uint8: caràcters BLANCS sobre fons NEGRE.
    """
    gray   = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        config.ADAPT_BLOCK,
        config.ADAPT_C,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_CLOSE_KERNEL)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
