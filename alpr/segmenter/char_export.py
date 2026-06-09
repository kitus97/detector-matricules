"""
alpr/segmenter/char_export.py
==============================
Fase 2, pas 5: retall i exportació dels caràcters a 28×28.

Dos formats seleccionables (`fmt`):

· "vj" (default) — retall TIGHT del binari de detecció + `resize 28×28 INTER_AREA`.
  Estira l'aspect ratio i omple vora a vora.

· "regla-or" — replica EXACTAMENT el procés d'entrenament de la CNN (la "REGLA
  D'OR"): glif en GRISOS centrat en un llenç 64×64 amb marge (AR preservat, com
  `01_render_fonts.py`) → binarització del "Grup D" (`adaptiveThreshold` MEAN_C/
  INV/31/15 a 64×64, com `02_augment.py`) → `resize 28×28 INTER_NEAREST`. La
  binarització es fa a l'escala de 64×64 (no a l'escala de detecció), de manera
  que el gruix de traç i la textura de vores coincideixin amb l'entrenament.
"""

import cv2
import numpy as np

from alpr import config


def preprocessa_caracter(
    thresh: np.ndarray,
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    fmt: str | None = None,
) -> np.ndarray:
    """
    Construeix la imatge 28×28 d'un caràcter (blanc sobre negre).

    Paràmetres
    ----------
    thresh : binari de detecció (caràcters blancs sobre negre) — font del format "vj".
    gray   : gris normalitzat i deskew-at (mateix espai de coords que thresh) —
             font del format "regla-or" (cal el glif en grisos per re-binaritzar).
    bbox   : (x, y, w, h) del caràcter.
    fmt    : "vj" (default) o "regla-or".

    Retorna np.ndarray 28×28 uint8 binari (blanc sobre negre).
    """
    if fmt is None:
        fmt = config.FORMAT_SORTIDA

    x, y, w, h = bbox

    # ── Format "vj": retall tight del binari + resize directe ─────────────────
    if fmt == "vj":
        crop = thresh[y : y + h, x : x + w]
        if crop.size == 0:
            return np.zeros((config.OUTPUT_SIZE, config.OUTPUT_SIZE), np.uint8)
        return cv2.resize(
            crop, (config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    # ── Format "regla-or": REGLA D'OR (render + Grup D exactes sobre grisos) ───
    if fmt == "regla-or":
        H, W = gray.shape[:2]
        # 1px de seguretat per no escapçar el contorn, sempre dins de la imatge
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(W, x + w + 1), min(H, y + h + 1)
        glif = gray[y0:y1, x0:x1]
        if glif.size == 0:
            return np.zeros((config.OUTPUT_SIZE, config.OUTPUT_SIZE), np.uint8)

        gh, gw = glif.shape[:2]
        objectiu = config.CANVAS_SIZE - 2 * config.MARGIN     # 52 px en la dim. gran
        escala = objectiu / float(max(gh, gw))
        nou_w = max(1, int(round(gw * escala)))
        nou_h = max(1, int(round(gh * escala)))
        interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
        glif_esc = cv2.resize(glif, (nou_w, nou_h), interpolation=interp)

        # Llenç BLANC i enganxat centrat (preserva l'AR; la dim. petita queda amb marge)
        llenc = np.full((config.CANVAS_SIZE, config.CANVAS_SIZE), 255, np.uint8)
        off_y = (config.CANVAS_SIZE - nou_h) // 2
        off_x = (config.CANVAS_SIZE - nou_w) // 2
        llenc[off_y:off_y + nou_h, off_x:off_x + nou_w] = glif_esc

        # Grup D EXACTE (a 64×64) → caràcter BLANC sobre fons NEGRE
        binar = cv2.adaptiveThreshold(
            llenc, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            config.ADAPT_BLOCK,
            config.ADAPT_C,
        )
        # Reducció final a 28×28 (últim pas, INTER_NEAREST com l'augmentation)
        return cv2.resize(
            binar, (config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

    raise ValueError(f"Format desconegut: '{fmt}'. Usa 'vj' o 'regla-or'.")


def crops_and_resize(
    thresh: np.ndarray,
    gray: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    fmt: str | None = None,
) -> list[np.ndarray]:
    """
    Aplica preprocessa_caracter a tots els bboxes.
    Retorna llista de np.ndarray 28×28.
    """
    return [preprocessa_caracter(thresh, gray, bbox, fmt) for bbox in bboxes]
