"""
alpr/segmenter/validate.py
===========================
Fase 2, pas 4: validació geomètrica del conjunt de caràcters detectats.

Comprova que els bboxes siguin coherents amb una matrícula real.
Llindars conservadors: millor rebutjar un crop dubtós que contaminar
el dataset d'entrenament amb caràcters espuris.
"""

import numpy as np

import config


def is_plausible_plate(
    bboxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,  # reservat per a futures comprovacions
) -> tuple[bool, str]:
    """
    Valida si els bboxes detectats corresponen a una matrícula real.

    Comprovacions:
      1. Recompte dins [N_CHARS_MIN, N_CHARS_MAX]
      2. Uniformitat d'alçades:     std_h  / h_med  <= ROW_STD_H_MAX
      3. Alineació de baseline:     std_cy / h_med  <= ROW_STD_CY_MAX
      4. Cobertura horitzontal:     (x_max - x_min) / img_w >= WIDTH_OCC_MIN

    Retorna (True, '') si vàlid, o (False, motiu) si rebutjat.
    """
    n = len(bboxes)
    if not (config.N_CHARS_MIN <= n <= config.N_CHARS_MAX):
        return False, f"recompte={n} fora de [{config.N_CHARS_MIN},{config.N_CHARS_MAX}]"

    heights = [h for (_, _, _, h) in bboxes]
    h_med   = float(np.median(heights))
    h_std   = float(np.std(heights))
    cys     = [y + h / 2.0 for (_, y, _, h) in bboxes]
    cy_std  = float(np.std(cys))
    x_min   = min(x       for (x, _, _, _) in bboxes)
    x_max   = max(x + w   for (x, _, w, _) in bboxes)
    x_cov   = (x_max - x_min) / img_w if img_w > 0 else 0.0

    if h_med > 0 and h_std > config.ROW_STD_H_MAX * h_med:
        return False, (
            f"alçades irregulars "
            f"(std/med={h_std/h_med:.0%} > {config.ROW_STD_H_MAX:.0%})"
        )
    if h_med > 0 and cy_std > config.ROW_STD_CY_MAX * h_med:
        return False, (
            f"mala alineació vertical "
            f"(cy_std/med={cy_std/h_med:.0%} > {config.ROW_STD_CY_MAX:.0%})"
        )
    if x_cov < config.WIDTH_OCC_MIN:
        return False, (
            f"amplada insuficient "
            f"({x_cov:.0%} < {config.WIDTH_OCC_MIN:.0%})"
        )
    return True, ""
