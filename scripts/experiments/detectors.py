"""
scripts/experiments/detectors.py
=================================
Detectors de matrícula COMPARABLES per a l'experiment de Fase 1
(memòria §4.1.3 "alternatives descartades" i §5.2 "resultats del detector").

Cada tècnica exposa la mateixa interfície:

    detect_boxes(img_bgr) -> list[(x, y, w, h)]

perquè totes s'avaluïn amb el mateix harness de recall (compare_detectors.py).

⚠️ Aquests detectors són NOMÉS per a la comparació experimental. El pla de
migració marca el morfològic com a oficial i la resta com a "referència, no camí
principal", per això NO viuen a `alpr/`:
  · morfologic       → reusa `alpr.detector` (única font de veritat de l'oficial).
  · canny_multiescala → còpia AUTOCONTINGUDA de l'algorisme de `main_3.py`. Es
    copia a propòsit (no s'importa) perquè `main_3.py` es retirarà en el futur;
    així aquest experiment seguirà funcionant quan aquell fitxer desaparegui.
    També s'ha tret la dependència d'`imutils`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Arrel del projecte (aquest fitxer viu a <arrel>/scripts/experiments/)
ARREL = Path(__file__).resolve().parents[2]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr.detector.detector import detect_boxes as _morpho_detect_boxes


# ══════════════════════════════════════════════════════════════════════════════
# Tècnica 1 — Morfològica (Sobel vertical) · OFICIAL (reusa alpr)
# ══════════════════════════════════════════════════════════════════════════════

def morphologic(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detector oficial de la Fase 1 (ADR-001). Delega a `alpr.detector`."""
    return _morpho_detect_boxes(img_bgr)


# ══════════════════════════════════════════════════════════════════════════════
# Tècnica 2 — Canny adaptatiu multi-escala  (còpia autocontinguda de main_3.py)
# ══════════════════════════════════════════════════════════════════════════════

_SCALES          = [1.0, 0.75, 0.50]
_CANNY_SIGMA     = 0.33
_AR_MIN, _AR_MAX = 1.5, 7.0
_POLY_MIN, _POLY_MAX     = 4, 6
_AREA_MIN_FRAC, _AREA_MAX_FRAC = 0.0005, 0.15
_EDGE_DENSITY_MIN = 0.03
_MAX_CANDIDATES   = 5
_NMS_IOU          = 0.4


def _grab_contours(found) -> list:
    """Equivalent a imutils.grab_contours: maneja el retorn de findContours
    (2 elements a OpenCV 4.x, 3 a 3.x) sense dependre d'imutils."""
    return found[0] if len(found) == 2 else found[1]


def _canny_adaptive(filtered: np.ndarray, sigma: float = _CANNY_SIGMA) -> np.ndarray:
    """Canny amb llindars adaptatius a la mediana de la imatge (autocalibració)."""
    median = float(np.median(filtered))
    t_low  = int(max(0,   (1.0 - sigma) * median))
    t_high = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(filtered, t_low, t_high)


def _otsu_morphology(gray: np.ndarray) -> np.ndarray:
    """Via alternativa (fallback): Otsu invers + closing horitzontal (25×5)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _edge_density(contour: np.ndarray, edges_ref: np.ndarray) -> float:
    """Densitat de vores Canny dins la bounding box del contorn."""
    x, y, w, h = cv2.boundingRect(contour)
    ih, iw = edges_ref.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    roi_area = (x2 - x1) * (y2 - y1)
    if roi_area == 0:
        return 0.0
    return np.count_nonzero(edges_ref[y1:y2, x1:x2]) / roi_area


def _iou_box(a: tuple, b: tuple) -> float:
    xA, yA, wA, hA = a
    xB, yB, wB, hB = b
    ix1, iy1 = max(xA, xB), max(yA, yB)
    ix2, iy2 = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = wA * hA + wB * hB - inter
    return inter / union if union > 0 else 0.0


def _nms(contours: list, iou_threshold: float = _NMS_IOU) -> list:
    """Non-Maximum Suppression sobre contorns (ordena per àrea, suprimeix solapats)."""
    if len(contours) <= 1:
        return contours
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = [w * h for (_, _, w, h) in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    kept, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(contours[i])
        for j in order:
            if j == i or j in suppressed:
                continue
            if _iou_box(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)
    return kept


def _extract_candidates(edges: np.ndarray, edges_ref: np.ndarray,
                        img_shape: tuple, scale: float) -> list:
    """Extreu contorns quasi-rectangulars validats (vèrtexs, AR, àrea, densitat)."""
    orig_h, orig_w = img_shape[:2]
    img_area = orig_h * orig_w

    found = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = _grab_contours(found)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    out = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if not (_POLY_MIN <= len(approx) <= _POLY_MAX):
            continue
        if scale != 1.0:
            approx = (approx / scale).astype(np.int32)
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        if not (_AR_MIN <= w / float(h) <= _AR_MAX):
            continue
        if not (_AREA_MIN_FRAC <= (w * h) / img_area <= _AREA_MAX_FRAC):
            continue
        if _edge_density(approx, edges_ref) < _EDGE_DENSITY_MIN:
            continue
        out.append(approx)
    return out


def canny_multiscale(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detector Canny adaptatiu multi-escala amb fallback Otsu+morfologia
    (algorisme de main_3.py). Retorna bounding boxes (x, y, w, h).
    """
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edges_ref = _canny_adaptive(filtered)

    orig_h, orig_w = img_bgr.shape[:2]
    all_cands: list = []
    for scale in _SCALES:
        if scale == 1.0:
            filtered_s = filtered
        else:
            filtered_s = cv2.resize(filtered, (int(orig_w * scale), int(orig_h * scale)),
                                    interpolation=cv2.INTER_AREA)
        edges_s = _canny_adaptive(filtered_s)
        all_cands.extend(_extract_candidates(edges_s, edges_ref, img_bgr.shape, scale))

    if not all_cands:   # Via B: Otsu + morfologia
        all_cands = _extract_candidates(_otsu_morphology(gray), edges_ref,
                                        img_bgr.shape, 1.0)

    final = _nms(all_cands)[:_MAX_CANDIDATES]
    return [tuple(int(v) for v in cv2.boundingRect(c)) for c in final]


# ══════════════════════════════════════════════════════════════════════════════
# Tècnica 3 — Cantonades (Harris + Shi-Tomasi) → mapa de densitat
#             (còpia autocontinguda de notebooks/fase_01/01_corner_detection_2.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

# Harris
_HK_K, _HK_KSIZE, _HK_THRESH = 0.04, 3, 0.001
# Shi-Tomasi
_ST_MAX, _ST_QUALITY, _ST_MIN_DIST = 600, 0.005, 3
# NMS sobre punts de cantonada (px)
_CORNER_NMS_RADIUS = 5
# Mapa de densitat
_DENSITY_SIGMA = 6
# Morfologia sobre el mapa binari
_DM_CLOSE_W, _DM_CLOSE_H = 15, 3
# Filtre de forma
_DM_AREA_MIN, _DM_AREA_MAX = 0.0005, 0.12
_DM_AR_MIN, _DM_AR_MAX     = 1.5, 9.0
_DM_MIN_W, _DM_MIN_H       = 20, 6
_DM_MIN_CONTOUR_AREA       = 60


def _corner_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Gris → bilateral(9,75,75) → CLAHE(2.0,(8,8))."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)


def _detect_harris(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=_HK_KSIZE, k=_HK_K)
    ys, xs = np.where(r > _HK_THRESH * r.max())
    return np.column_stack([xs, ys]).astype(np.float32), r[ys, xs]


def _detect_shi_tomasi(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = cv2.goodFeaturesToTrack(gray, _ST_MAX, _ST_QUALITY, _ST_MIN_DIST)
    if p is None:
        return np.empty((0, 2), np.float32), np.empty(0)
    pts = p.reshape(-1, 2)
    return pts, np.ones(len(pts))


def _nms_corners(pts: np.ndarray, scores: np.ndarray,
                 radius: int = _CORNER_NMS_RADIUS) -> np.ndarray:
    """Suprimeix cantonades dins d'un radi, prioritzant les de més score."""
    if len(pts) == 0:
        return pts
    order = np.argsort(scores)[::-1]
    pts = pts[order]
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dists = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        keep[i + 1:][dists < radius] = False
    return pts[keep]


def _merge_corners(gray: np.ndarray) -> np.ndarray:
    """Harris + Shi-Tomasi, cadascun amb NMS, fusionats."""
    ph, sh = _detect_harris(gray);     ph = _nms_corners(ph, sh)
    ps, ss = _detect_shi_tomasi(gray); ps = _nms_corners(ps, ss)
    if len(ph) > 0 and len(ps) > 0:
        return np.vstack([ph, ps])
    return ph if len(ph) > 0 else ps


def _build_density_map(pts: np.ndarray, img_shape: tuple,
                       sigma: float = _DENSITY_SIGMA) -> np.ndarray:
    """Acumula els punts de cantonada i els suavitza amb un Gaussià → mapa de densitat."""
    H, W = img_shape[:2]
    d = np.zeros((H, W), dtype=np.float32)
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            d[yi, xi] += 1.0
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(d, (ksize, ksize), sigmaX=sigma)


def _density_to_boxes(density: np.ndarray, img_shape: tuple) -> list[tuple[int, int, int, int]]:
    """Normalitza → Otsu → closing horitzontal + opening → contorns → filtre de forma."""
    H, W = img_shape[:2]
    img_area = H * W
    d8 = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_DM_CLOSE_W, _DM_CLOSE_H)))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in _grab_contours(found):
        if cv2.contourArea(cnt) < _DM_MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < _DM_MIN_W or h < _DM_MIN_H:
            continue
        if not (_DM_AREA_MIN <= w * h / img_area <= _DM_AREA_MAX):
            continue
        if not (_DM_AR_MIN <= w / float(h) <= _DM_AR_MAX):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def harris_density(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detector per CANTONADES: Harris + Shi-Tomasi → mapa de densitat de cantonades
    → Otsu + morfologia → bounding boxes. La matrícula concentra moltes cantonades
    (caràcters) → un pic de densitat. Retorna list[(x, y, w, h)].
    """
    enhanced = _corner_preprocess(img_bgr)
    pts      = _merge_corners(enhanced)
    density  = _build_density_map(pts, img_bgr.shape)
    return _density_to_boxes(density, img_bgr.shape)


# ══════════════════════════════════════════════════════════════════════════════
# Registre de detectors comparables
# ══════════════════════════════════════════════════════════════════════════════

DETECTORS = {
    "morfologic":        morphologic,
    "canny_multiescala": canny_multiscale,
    "harris_density":    harris_density,
}
