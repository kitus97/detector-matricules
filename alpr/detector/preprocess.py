"""
alpr/detector/preprocess.py
============================
Fase 1, pas 1: preprocessament de la imatge per al detector.
Grisos → Gaussian → CLAHE.
"""

import cv2
import numpy as np

import config


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Converteix a grisos, suavitza amb Gaussian i equalitza el contrast amb CLAHE.

    Retorna imatge 1-canal uint8 lista per a l'extracció de vores.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=config.GAUSS_KSIZE, sigmaX=config.GAUSS_SIGMA)
    clahe   = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_SIZE,
    )
    return clahe.apply(blurred)
