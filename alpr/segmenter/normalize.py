"""
alpr/segmenter/normalize.py
============================
Fase 2, pas 0 (previ a la detecció): normalització d'escala del crop.

Per què (D9 / ADR-002): les plaques d'aquest dataset són molt petites
(alçada GT de 18–46 px) i el `blockSize=31` de la binarització adaptativa està
calibrat per a una escala concreta. Si es binaritza el crop a la seva mida
original, el bloc és més gran que el caràcter (o que el crop sencer) i la
binarització falla → sobre-segmentació. Normalitzant tots els crops a una
alçada de treball comuna (`NORM_H`) abans de detectar, el `blockSize` té sempre
una escala coherent respecte als caràcters. Com a efecte secundari positiu,
upscalar caràcters minúsculs millora la qualitat del 28×28 final.
"""

import cv2
import numpy as np

from alpr import config


def normalitza_escala(img: np.ndarray, target_h: int | None = None) -> np.ndarray:
    """
    Redimensiona la imatge a una alçada de treball comuna preservant l'aspect ratio.

    Funciona tant amb imatges BGR (3 canals) com en escala de grisos: només
    afecta les dimensions espacials. Usa INTER_CUBIC per ampliar (recupera
    detall en plaques minúscules) i INTER_AREA per reduir (evita aliasing).

    Args:
        img:      imatge d'entrada (BGR uint8 o gris), qualsevol mida.
        target_h: alçada objectiu en px; si és None, s'usa config.NORM_H.

    Returns:
        La imatge redimensionada a target_h d'alçada (mateixa que l'entrada si
        ja hi coincideix).
    """
    if target_h is None:
        target_h = config.NORM_H

    h, w = img.shape[:2]
    if h == target_h:
        return img

    new_w  = max(1, int(round(w * target_h / float(h))))
    interp = cv2.INTER_CUBIC if h < target_h else cv2.INTER_AREA
    return cv2.resize(img, (new_w, target_h), interpolation=interp)
