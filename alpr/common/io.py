"""
alpr/common/io.py
=================
Utilitats bàsiques d'entrada/sortida d'imatges.
"""

from pathlib import Path

import cv2
import numpy as np

import config


def load_image(path: Path | str) -> np.ndarray:
    """
    Carrega una imatge en format BGR uint8.
    Llança FileNotFoundError si no es troba o ValueError si no es pot llegir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Imatge no trobada: '{path}'")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No s'ha pogut llegir la imatge: '{path}'")
    return img


def save_image(img: np.ndarray, path: Path | str) -> None:
    """Guarda una imatge BGR. Crea els directoris pares si cal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Converteix BGR → grisos (1 canal)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def iter_images(directory: Path | str):
    """
    Generador que retorna (path, img_bgr) per a totes les imatges
    d'un directori (extensió a config.IMAGE_EXTENSIONS).
    Les imatges que no es puguin llegir s'ometen amb un avís.
    """
    import logging
    log = logging.getLogger(__name__)

    directory = Path(directory)
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    for p in paths:
        try:
            yield p, load_image(p)
        except (FileNotFoundError, ValueError) as e:
            log.warning(str(e))
