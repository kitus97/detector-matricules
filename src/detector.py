"""
Etapa A — Generació ràpida de candidats a matrícula.

Pipeline (visió per computador clàssica, sense deep learning):
    1. Preprocessament (gris + CLAHE).
    2. Gradient vertical de Sobel + binarització Otsu.
    3. Clausura morfològica amb kernel rectangular horitzontal.
    4. Components connexos -> bounding boxes.
    5. Filtrat per propietats geomètriques (aspect ratio, àrea, solidesa, mida).
    6. Sortida: llista de ROIs + visualitzacions de debug.

Funció principal exposada:
    generate_plate_candidates(image_path_or_dir, verbose=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# =============================================================================
# CONSTANTS DE CALIBRATGE
# =============================================================================
# Totes agrupades aquí per facilitar el tuning sense buscar pel codi.

# --- CLAHE -------------------------------------------------------------------
# clip_limit moderat (2.0): si pugem massa, el soroll es dispara a fons uniformes.
# tile (8x8): equilibri entre adaptació local i evitar artefactes en blocs.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# --- Sobel -------------------------------------------------------------------
# Gradient VERTICAL (dx=1, dy=0) perquè els caràcters d'una matrícula generen
# moltes vores verticals fortes (transicions clar↔fosc al llarg de l'eix X).
SOBEL_KSIZE: int = 3

# --- Morfologia --------------------------------------------------------------
# Kernel rectangular ~ (20 x 3): "fusiona" caràcters propers horitzontalment
# (la separació entre caràcters d'una matrícula sol ser de pocs píxels) sense
# unir-los amb zones de soroll situades verticalment a sobre/sota de la placa.
MORPH_KERNEL_SIZE: tuple[int, int] = (20, 3)
# Petita obertura final per netejar punts aïllats que no formen blob real.
MORPH_OPEN_KERNEL_SIZE: tuple[int, int] = (3, 3)

# --- Filtrat geomètric -------------------------------------------------------
# Aspect ratio (w/h):
#   - Matrícules europees ronden 4.5:1 (520x110 mm).
#   - Matrícules USA ronden 2:1 (300x150 mm).
#   - Permetem 2.0–6.0 per cobrir inclinacions lleus i variabilitat de tipus.
ASPECT_RATIO_MIN: float = 2.0
ASPECT_RATIO_MAX: float = 6.0

# Àrea relativa respecte la imatge sencera:
#   - 0.05% filtra components diminuts (soroll, segells, lletres soltes).
#   - 5% filtra panells, finestres, ombres grans.
AREA_RATIO_MIN: float = 0.0005   # 0.05 %
AREA_RATIO_MAX: float = 0.05     # 5    %

# Solidesa (area / convex_hull_area):
#   - Una matrícula és quasi rectangular -> solidity proper a 1.
#   - >0.5 elimina blobs irregulars (fulles, branques, reflexos).
SOLIDITY_MIN: float = 0.5

# Dimensions mínimes en píxels:
#   - <20 px d'amplada no és OCR-viable i sol ser soroll.
MIN_WIDTH_PX: int = 20
MIN_HEIGHT_PX: int = 8


# =============================================================================
# PIPELINE
# =============================================================================

def preprocess_image(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converteix a gris i aplica CLAHE.

    Retorna (gray, gray_clahe) per poder visualitzar tots dos.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    gray_eq = clahe.apply(gray)
    return gray, gray_eq


def detect_edges(gray_eq: np.ndarray) -> np.ndarray:
    """Gradient vertical de Sobel + Otsu sobre la magnitud."""
    # dx=1, dy=0 => derivada en X => respon a vores VERTICALS (transicions
    # horitzontals d'intensitat). És el patró dominant dels caràcters.
    sobel_x = cv2.Sobel(gray_eq, cv2.CV_16S, dx=1, dy=0, ksize=SOBEL_KSIZE)
    mag = cv2.convertScaleAbs(sobel_x)

    # Otsu sobre la magnitud: triem el llindar automàticament segons l'histograma
    # global. És robust si la imatge té contrast raonable post-CLAHE.
    _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """Closing horitzontal per fusionar caràcters + opening per netejar."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL_SIZE)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return cleaned


def find_candidates(morph: np.ndarray, image_shape: tuple[int, int]) -> list[dict]:
    """Components connexos -> bounding boxes -> filtrat geomètric."""
    h_img, w_img = image_shape[:2]
    img_area = float(h_img * w_img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        morph, connectivity=8
    )

    candidates: list[dict] = []
    # Saltem label 0 (fons).
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if w < MIN_WIDTH_PX or h < MIN_HEIGHT_PX:
            continue

        aspect = w / float(h) if h > 0 else 0.0
        if not (ASPECT_RATIO_MIN <= aspect <= ASPECT_RATIO_MAX):
            continue

        area_ratio = area / img_area
        if not (AREA_RATIO_MIN <= area_ratio <= AREA_RATIO_MAX):
            continue

        # Solidesa: necessita el contorn del blob per calcular el convex hull.
        component_mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = (cv2.contourArea(cnt) / hull_area) if hull_area > 0 else 0.0
        if solidity < SOLIDITY_MIN:
            continue

        candidates.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "area": int(area),
            "aspect_ratio": float(aspect),
            "solidity": float(solidity),
        })

    return candidates


def draw_candidates(bgr: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Dibuixa les ROIs sobre una còpia de la imatge original."""
    out = bgr.copy()
    for c in candidates:
        cv2.rectangle(out, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)
    return out


# =============================================================================
# VISUALITZACIÓ DE DEBUG
# =============================================================================

def _show_debug(
    title: str,
    bgr: np.ndarray,
    gray_eq: np.ndarray,
    edges: np.ndarray,
    morph: np.ndarray,
    overlay: np.ndarray,
) -> None:
    """Mosaic 2x3 amb les etapes intermèdies. Usa matplotlib (no bloquejant)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(title)

    axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(gray_eq, cmap="gray")
    axes[0, 1].set_title("Gris + CLAHE")
    axes[0, 2].imshow(edges, cmap="gray")
    axes[0, 2].set_title("Sobel-X + Otsu")
    axes[1, 0].imshow(morph, cmap="gray")
    axes[1, 0].set_title(f"Closing {MORPH_KERNEL_SIZE}")
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Candidats finals")
    axes[1, 2].axis("off")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# =============================================================================
# API PÚBLICA
# =============================================================================

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(path: str | Path) -> Iterable[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in _IMG_EXT:
                yield f
    else:
        raise FileNotFoundError(f"Path no existent: {path}")


def _process_one(image_path: Path, verbose: bool) -> list[dict]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        if verbose:
            print(f"[WARN] No s'ha pogut llegir: {image_path}")
        return []

    _, gray_eq = preprocess_image(bgr)
    edges = detect_edges(gray_eq)
    morph = apply_morphology(edges)
    candidates = find_candidates(morph, bgr.shape)

    if verbose:
        print(f"[{image_path.name}] candidats: {len(candidates)}")
        overlay = draw_candidates(bgr, candidates)
        _show_debug(image_path.name, bgr, gray_eq, edges, morph, overlay)

    return candidates


def generate_plate_candidates(
    image_path_or_dir: str | os.PathLike,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Genera ROIs candidates a matrícula per a una imatge o un directori.

    Args:
        image_path_or_dir: ruta a una imatge o a un directori amb imatges.
        verbose: si True, imprimeix info i mostra visualitzacions de debug.

    Returns:
        dict {nom_fitxer: [ {x1,y1,x2,y2,area,aspect_ratio,solidity}, ... ]}
    """
    results: dict[str, list[dict]] = {}
    for img_path in _iter_images(image_path_or_dir):
        results[img_path.name] = _process_one(img_path, verbose=verbose)
    return results
