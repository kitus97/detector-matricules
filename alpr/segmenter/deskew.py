"""
alpr/segmenter/deskew.py
=========================
Fase 2, pas 1: correcció d'inclinació (deskew) del crop de matrícula.

Mètode: **regressió lineal sobre els centroides** dels caràcters candidats
(ADR-003). Els caràcters d'una matrícula estan alineats horitzontalment per
disseny, així que el pendent de la recta que passa pels seus centroides revela
directament la inclinació de la placa — un senyal molt més net que la forma
global del crop.

> Per què NO `minAreaRect` global: es va implementar i descartar (ADR-003). El
> núvol global de punts de primer pla (fons binaritzat, vores de la placa,
> soroll) forma una forma aproximadament rectangular **alineada amb els eixos**,
> de manera que l'angle resultant és gairebé sempre ≈ 0° independentment de la
> inclinació real.

Esquema de dues passades: aquesta funció fa la **passada 1 tolerant** (detecta
candidats només per estimar l'angle); la **passada 2 estricta** (segmentació
final) la fa `segment()` sobre el crop ja alineat.
"""

import cv2
import numpy as np

from alpr import config
from .binarize import binarize_adaptive
from .contours import extract_contours, filter_geometric


def estimate_skew_angle(bboxes: list[tuple[int, int, int, int]]) -> float:
    """
    Estima l'angle d'inclinació (graus) per regressió sobre els centroides.

    Ajusta una recta `cy = m·cx + b` als centroides dels bboxes candidats i
    retorna `arctan(m)` en graus. Retorna 0.0 si:
      - hi ha menys de DESKEW_MIN_CANDIDATS punts (senyal insuficient), o
      - `|angle| > DESKEW_ANGLE_MAX` (es considera detecció errònia → no rotar).
    """
    if len(bboxes) < config.DESKEW_MIN_CANDIDATS:
        return 0.0

    cx = np.array([x + w / 2.0 for (x, _, w, _) in bboxes])
    cy = np.array([y + h / 2.0 for (_, y, _, h) in bboxes])
    slope, _ = np.polyfit(cx, cy, 1)
    angle = float(np.degrees(np.arctan(slope)))

    if abs(angle) > config.DESKEW_ANGLE_MAX:
        return 0.0
    return angle


def deskew(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Corregeix la inclinació d'un crop de matrícula (passada 1: tolerant).

    Detecta caràcters candidats amb la binarització adaptativa i el filtre
    geomètric, n'estima l'angle per regressió de centroides i rota el crop al
    voltant del seu centre. BORDER_REPLICATE evita franges negres als cantons.

    Retorna (aligned_bgr, angle_graus). angle_graus = 0.0 si no s'ha rotat.
    """
    thresh    = binarize_adaptive(crop_bgr)
    bboxes    = extract_contours(thresh)
    # Passada 1: filtre TOLERANT (strict=False) relatiu a l'alçada del crop, només
    # per obtenir prou candidats per estimar l'angle.
    candidats = filter_geometric(bboxes, crop_bgr.shape[0], strict=False)
    angle     = estimate_skew_angle(candidats)

    if abs(angle) < 1e-3:
        return crop_bgr, 0.0

    H_img, W_img = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W_img / 2.0, H_img / 2.0), angle, 1.0)
    aligned = cv2.warpAffine(
        crop_bgr, M, (W_img, H_img),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, angle
