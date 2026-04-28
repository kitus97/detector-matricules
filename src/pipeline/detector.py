"""
Etapa A — Generació ràpida de candidats a matrícula.

Pipeline (visió per computador clàssica, sense deep learning):
    1. Preprocessament (gris + CLAHE).
    2. Blur gaussià + Sobel-X + llindar per percentil.
    3. Projecció horitzontal H(y) -> bandes candidates.
    4. Per cada banda, projecció vertical V(x) -> segments candidats.
    5. Filtrat geomètric (aspect ratio, àrea relativa).
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
# Gradient vertical (dx=1, dy=0) perquè els caràcters d'una matrícula generen
# moltes vores verticals fortes (transicions clar↔fosc al llarg de l'eix X).
SOBEL_KSIZE: int = 3
# Blur gaussià pre-Sobel: atenua píxels-soroll que d'altra manera disparen el
# percentil cap amunt i deixarien fora text de matrícula amb gradient mitjà.
SOBEL_BLUR_KSIZE: int = 5

# --- Llindar per percentil ---------------------------------------------------
# El 70è percentil de la magnitud Sobel positiva: més estable que Otsu davant
# escenes amb grans àrees uniformes (cel, carrosseria) que esbiaixen Otsu.
EDGE_PERCENTILE: float = 90.0

# --- Bandes de files (projecció H(y)) ---------------------------------------
# Llindar sobre H_norm: 0.6 deixa passar bandes amb densitat moderada de vores.
H_THRESHOLD: float = 0.6
# Mida mínima d'una banda; per sota són pics aïllats, no text.
MIN_BAND_HEIGHT: int = 8
# Mida màxima d'una banda relativa a l'alçada de la imatge; per sobre és fons
# (rejes de radiador, ombres llargues, etc.).
MAX_BAND_HEIGHT_RATIO: float = 0.25

# --- Segments de columnes (projecció V(x)) ----------------------------------
V_THRESHOLD: float = 0.10
# Gap màxim a fusionar (relatiu a l'amplada): permet unir caràcters separats
# per espais petits dins la mateixa matrícula.
COL_GAP_RATIO: float = 0.025
# Amplada mínima d'un candidat respecte la imatge: per sota no és OCR-viable.
MIN_PLATE_WIDTH_RATIO: float = 0.05

# --- Filtrat geomètric final -------------------------------------------------
# Marge ampli (1.5–7.0): prioritzem recall; la precisió la millorarà l'Etapa B.
ASPECT_RATIO_MIN: float = 1.5
ASPECT_RATIO_MAX: float = 7.0

# Àrea relativa: 0.2% filtra micro-candidats, 15% filtra panells/finestres.
AREA_RATIO_MIN: float = 0.002
AREA_RATIO_MAX: float = 0.20


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
    """Sobel-X + llindar per percentil.

    Aplica un blur gaussià abans del Sobel per reduir soroll puntual, i després
    binaritza prenent com a llindar el percentil EDGE_PERCENTILE de la
    magnitud positiva. Si la imatge és plana (cap píxel positiu), retorna
    binari tot a zero — la resta del pipeline ho gestiona retornant 0 candidats.
    """
    blurred = cv2.GaussianBlur(gray_eq, (SOBEL_BLUR_KSIZE, SOBEL_BLUR_KSIZE), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, dx=1, dy=0, ksize=SOBEL_KSIZE)
    mag = cv2.convertScaleAbs(sobel_x)
    positives = mag[mag > 0]
    if positives.size == 0:
        return np.zeros_like(mag)
    threshold = float(np.percentile(positives, EDGE_PERCENTILE))
    binary = (mag >= threshold).astype(np.uint8) * 255
    return binary


def _group_consecutive(
    indices: np.ndarray, max_gap: int = 1
) -> list[tuple[int, int]]:
    """Agrupa índexs creixents en bandes (start, end) inclusives.

    Dos índexs consecutius pertanyen a la mateixa banda si la seva distància
    és <= max_gap. Útil per extreure trams contigus d'un vector binari de
    "files/columnes actives".
    """
    if indices.size == 0:
        return []
    bands: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        i = int(idx)
        if i - prev <= max_gap:
            prev = i
        else:
            bands.append((start, prev))
            start = i
            prev = i
    bands.append((start, prev))
    return bands


def find_candidate_rows(
    binary: np.ndarray, image_height: int
) -> list[tuple[int, int]]:
    """Bandes (y_start, y_end) inclusives amb alta densitat horitzontal de vores."""
    H = np.sum(binary > 0, axis=1).astype(np.float32)
    if H.max() == 0:
        return []
    H_norm = H / np.median(H[H > 0])  # normalitza pel percentil 50 per estabilitat davant fons amb moltes vores
    H_smooth = np.convolve(H_norm, np.ones(5) / 5, mode="same")
    active = np.where(H_smooth > H_THRESHOLD)[0]
    if active.size == 0:
        return []
    bands = _group_consecutive(active, max_gap=1)
    max_band = int(image_height * MAX_BAND_HEIGHT_RATIO)
    return [
        (s, e) for (s, e) in bands if MIN_BAND_HEIGHT <= (e - s + 1) <= max_band
    ]


def find_candidate_cols(
    binary_strip: np.ndarray, image_width: int
) -> list[tuple[int, int]]:
    """Segments (x_start, x_end) inclusius dins una franja horitzontal."""
    V = np.sum(binary_strip > 0, axis=0).astype(np.float32)
    if V.max() == 0:
        return []
    V_norm = V / V.max()
    V_smooth = np.convolve(V_norm, np.ones(5) / 5, mode="same")
    active = np.where(V_smooth > V_THRESHOLD)[0]
    if active.size == 0:
        return []
    gap = max(1, int(image_width * COL_GAP_RATIO))
    segments = _group_consecutive(active, max_gap=gap)
    min_w = int(image_width * MIN_PLATE_WIDTH_RATIO)
    return [(s, e) for (s, e) in segments if (e - s + 1) >= min_w]


def build_candidates(
    row_bands: list[tuple[int, int]],
    binary: np.ndarray,
    image_shape: tuple[int, ...],
) -> list[dict]:
    """Combina bandes de files amb segments de columnes i filtra per geometria.

    Per a cada banda calculem el seu V(x) propi (les bandes diferents tenen
    distribucions de columnes diferents); després apliquem aspect ratio i àrea.
    Convenció bbox: (x1, y1) inclusius, (x2, y2) exclusius — coincideix amb el
    contracte previ del detector i amb com cv2.rectangle els consumeix.
    """
    h_img, w_img = image_shape[:2]
    img_area = float(h_img * w_img)

    candidates: list[dict] = []
    for (y_start, y_end) in row_bands:
        strip = binary[y_start:y_end + 1, :]
        col_segments = find_candidate_cols(strip, w_img)
        for (x_start, x_end) in col_segments:
            w = x_end - x_start + 1
            h = y_end - y_start + 1
            if h <= 0:
                continue
            ar = w / float(h)
            if not (ASPECT_RATIO_MIN <= ar <= ASPECT_RATIO_MAX):
                continue
            area = w * h
            area_ratio = area / img_area
            if not (AREA_RATIO_MIN <= area_ratio <= AREA_RATIO_MAX):
                continue
            candidates.append({
                "x1": int(x_start),
                "y1": int(y_start),
                "x2": int(x_end + 1),
                "y2": int(y_end + 1),
                "area": int(area),
                "aspect_ratio": float(ar),
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
    overlay: np.ndarray,
    row_bands: list[tuple[int, int]],
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
    axes[0, 2].set_title(f"Sobel-X + percentil {int(EDGE_PERCENTILE)}")

    ax_h = axes[1, 0]
    H = np.sum(edges > 0, axis=1).astype(np.float32)
    if H.max() > 0:
        H_norm = H / H.max()
    else:
        H_norm = H
    H_smooth = np.convolve(H_norm, np.ones(5) / 5, mode="same")
    ys = np.arange(len(H_smooth))
    ax_h.plot(H_smooth, ys, color="black", linewidth=0.8)
    ax_h.axvline(H_THRESHOLD, color="gray", linestyle="--", linewidth=0.5)
    for (y_s, y_e) in row_bands:
        ax_h.axhspan(y_s, y_e, color="red", alpha=0.3)
    ax_h.invert_yaxis()
    ax_h.set_xlim(0, 1.05)
    ax_h.set_ylim(len(H_smooth) - 1, 0)
    ax_h.set_title("H(y) amb bandes")

    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Candidats finals")
    axes[1, 2].axis("off")

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
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
    row_bands = find_candidate_rows(edges, bgr.shape[0])
    candidates = build_candidates(row_bands, edges, bgr.shape)

    if verbose:
        print(f"[{image_path.name}] candidats: {len(candidates)}")
        overlay = draw_candidates(bgr, candidates)
        _show_debug(image_path.name, bgr, gray_eq, edges, overlay, row_bands)

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
        dict {nom_fitxer: [ {x1,y1,x2,y2,area,aspect_ratio}, ... ]}
    """
    results: dict[str, list[dict]] = {}
    for img_path in _iter_images(image_path_or_dir):
        results[img_path.name] = _process_one(img_path, verbose=verbose)
    return results
