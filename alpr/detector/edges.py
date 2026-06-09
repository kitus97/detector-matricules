"""
alpr/detector/edges.py
======================
Fase 1, pas 2: Sobel vertical + binarització Otsu.

Els caràcters d'una matrícula generen una densitat anormalment alta de vores
verticals en una regió molt acotada. Explotant aquesta propietat estructural
podem localitzar la matrícula sense model entrenat.
"""

import cv2
import numpy as np


def sobel_vertical_binary(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica el filtre de Sobel vertical (dx=1, dy=0, ksize=3) i binaritza
    amb el mètode d'Otsu.

    Retorna (sobel_abs, binary):
      - sobel_abs : magnitud de gradient uint8
      - binary    : imatge binaritzada {0, 255}
    """
    sobel_x   = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel_x)
    _, binary = cv2.threshold(
        sobel_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return sobel_abs, binary
