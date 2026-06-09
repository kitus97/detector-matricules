"""
alpr/segmenter/projection.py
=============================
Fase 2 — mètode de segmentació ALTERNATIU: projecció vertical.

Mètode secundari (l'oficial és la detecció de contorns, ADR-002). Es conserva per
poder **comparar solucions**: sobre el binari de detecció, suma els píxels blancs
per columna; les bandes de columnes "amb tinta" separades per valls són els
caràcters. No requereix conèixer el nombre de caràcters a priori.

Mesura de referència: 40 ROIs acceptades (projecció) vs 79 (contorns) sobre els
mateixos 341 crops → els contorns són netament superiors en aquest dataset.
"""

import numpy as np

from alpr import config


def _bbox_de_banda(thresh: np.ndarray, x0: int, x1: int):
    """Donada una banda de columnes [x0, x1), retorna el bbox vertical del seu contingut."""
    banda = thresh[:, x0:x1]
    files = np.where(banda.sum(axis=1) > 0)[0]
    if len(files) == 0:
        return None
    y0, y1 = int(files.min()), int(files.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def detecta_projeccio(thresh: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detecta caràcters per projecció vertical sobre el binari (blanc sobre negre).

    Suavitza el histograma de columnes amb una mitjana mòbil, considera "amb tinta"
    les columnes per sobre d'una fracció del màxim, i extreu un bbox per cada banda
    contínua de columnes amb tinta. Retorna list[(x, y, w, h)].
    """
    col = (thresh > 0).sum(axis=0).astype(np.float64)
    if col.max() <= 0:
        return []

    kernel = np.ones(config.PROJ_SMOOTH_WIN) / config.PROJ_SMOOTH_WIN
    col_s = np.convolve(col, kernel, mode="same")
    llindar = max(config.PROJ_INK_FRAC * col_s.max(), 1.0)
    amb_tinta = col_s > llindar

    bboxes: list[tuple[int, int, int, int]] = []
    inici = None
    for i, val in enumerate(amb_tinta):
        if val and inici is None:
            inici = i
        elif not val and inici is not None:
            bb = _bbox_de_banda(thresh, inici, i)
            if bb is not None:
                bboxes.append(bb)
            inici = None
    if inici is not None:
        bb = _bbox_de_banda(thresh, inici, thresh.shape[1])
        if bb is not None:
            bboxes.append(bb)
    return bboxes
