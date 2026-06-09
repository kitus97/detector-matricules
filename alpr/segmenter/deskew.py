"""
alpr/segmenter/deskew.py
=========================
Fase 2, pas 0: correcció d'inclinació del crop de matrícula.

Usa minAreaRect sobre els píxels blancs de la binarització Otsu.
Si l'angle supera DESKEW_ANGLE_MAX no rota (probablement soroll).
"""

import cv2
import numpy as np

from alpr import config


def deskew(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Corregeix la inclinació d'un crop de matrícula.

    Retorna (aligned_bgr, angle_graus).
    angle_graus = 0.0 si no s'ha rotat.
    """
    gray      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Assegura que els caràcters siguin blancs
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    points = np.column_stack(np.where(binary == 255))
    if len(points) < 5:
        return crop_bgr, 0.0

    rect = cv2.minAreaRect(points[:, ::-1].astype(np.float32))
    (_, _), (w, h), angle = rect
    angle_corr = angle + 90.0 if w < h else angle

    if abs(angle_corr) > config.DESKEW_ANGLE_MAX:
        return crop_bgr, 0.0

    H_img, W_img = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W_img / 2.0, H_img / 2.0), angle_corr, 1.0)
    aligned = cv2.warpAffine(
        crop_bgr, M, (W_img, H_img),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, angle_corr
