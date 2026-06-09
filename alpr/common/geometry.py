"""
alpr/common/geometry.py
=======================
Utilitats geomètriques transversals: IoU i NMS.
"""


def iou(a: tuple, b: tuple) -> float:
    """
    Intersection over Union de dos bboxes (x, y, w, h).
    Retorna 0.0 si la unió és zero.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix    = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy    = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: list[tuple],
    scores: list[float] | None = None,
    iou_thresh: float = 0.5,
) -> list[tuple]:
    """
    Non-Maximum Suppression sobre una llista de bboxes (x, y, w, h).

    Si no es passen scores, s'ordena per àrea descendent (box més gran = prioritat).
    Retorna la llista de boxes retingudes ordenada per score descendent.
    """
    if not boxes:
        return []

    if scores is None:
        scores = [b[2] * b[3] for b in boxes]  # àrea com a proxy

    # Ordena per score descendent
    ranked = sorted(zip(scores, boxes), key=lambda x: -x[0])
    kept: list[tuple] = []

    for _score, box in ranked:
        if not any(iou(box, k) > iou_thresh for k in kept):
            kept.append(box)

    return kept
