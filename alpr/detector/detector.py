"""
alpr/detector/detector.py
==========================
API pública del detector de matrícules (Fase 1).

Encadena les quatre fases del pipeline morfològic:
  preprocess → sobel_vertical_binary → morphology_and_label → filter_by_shape

API pública:
  detect_boxes(img_bgr) -> list[(x, y, w, h)]
  detect(img_bgr)       -> list[np.ndarray]       retalls BGR
  detect_debug(img_bgr) -> dict                   totes les etapes intermèdies
"""

import numpy as np

from .preprocess   import preprocess
from .edges        import sobel_vertical_binary
from .morphology   import morphology_and_label
from .shape_filter import filter_by_shape


def detect_boxes(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detecta bboxes de candidats a matrícula.

    Retorna list[(x, y, w, h)] en píxels de la imatge original.
    Llista buida si no es detecta res.
    """
    enhanced                         = preprocess(img_bgr)
    _, binary                        = sobel_vertical_binary(enhanced)
    _, num_labels, _, stats          = morphology_and_label(binary)
    return filter_by_shape(num_labels, stats, img_bgr.shape)


def detect(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Detecta matrícules i retorna els retalls BGR de cada candidat.

    Retorna list[np.ndarray] (BGR uint8). Llista buida si no es detecta res.
    """
    boxes = detect_boxes(img_bgr)
    return [img_bgr[y : y + h, x : x + w] for (x, y, w, h) in boxes]


def detect_debug(img_bgr: np.ndarray) -> dict:
    """
    Com detect_boxes però retorna un dict amb totes les etapes intermèdies,
    útil per a diagnòstic i visualització.

    Claus del dict:
      enhanced  : imatge post-CLAHE
      sobel     : magnitud de gradient Sobel vertical
      binary    : binaritzada Otsu
      morph     : post-morfologia
      boxes     : list[(x, y, w, h)]
      n_total   : nombre de components connexos (sense el fons)
    """
    enhanced                              = preprocess(img_bgr)
    sobel_abs, binary                     = sobel_vertical_binary(enhanced)
    morph, num_labels, _labels, stats     = morphology_and_label(binary)
    boxes                                 = filter_by_shape(num_labels, stats, img_bgr.shape)

    return {
        "enhanced": enhanced,
        "sobel":    sobel_abs,
        "binary":   binary,
        "morph":    morph,
        "boxes":    boxes,
        "n_total":  num_labels - 1,
    }
