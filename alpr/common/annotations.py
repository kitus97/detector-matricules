"""
alpr/common/annotations.py
===========================
Parseig d'anotacions GT i noms de fitxer del projecte.

Format GT per línia:
    nom_imatge  x  y  w  h  MATRICULA   (separats per tabulador o espais)
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Patrons de nom de fitxer de crop i de caràcter segmentat
_RE_CROP = re.compile(r"^(.+)_(?:cand|box)(\d+)$")
_RE_CHAR = re.compile(r"^(.+)_box(\d+)_char(\d+)\.png$")


def get_txt_path(img_path: Path) -> Path:
    """Retorna la ruta del fitxer GT .txt corresponent a img_path."""
    return img_path.with_suffix(".txt")


def parse_annotation(line: str) -> dict | None:
    """
    Parseja una línia d'anotació GT.
    Retorna {'filename', 'x', 'y', 'w', 'h', 'plate'} o None si la línia és malformada.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 6:
        log.warning(f"Línia GT malformada: {line!r}")
        return None
    return {
        "filename": parts[0],
        "x":        int(parts[1]),
        "y":        int(parts[2]),
        "w":        int(parts[3]),
        "h":        int(parts[4]),
        "plate":    parts[5].upper(),
    }


def load_annotations(gt_path: Path) -> list[dict]:
    """
    Carrega totes les anotacions d'un fitxer GT.
    Retorna llista de dicts (veure parse_annotation).
    """
    if not gt_path.exists():
        return []
    results = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        ann = parse_annotation(line)
        if ann:
            results.append(ann)
    return results


def parse_crop_filename(path: Path) -> tuple[str | None, int | None]:
    """
    Extreu (stem_base, idx) de noms '{stem}_cand{n}' o '{stem}_box{n}'.
    Retorna (None, None) si el nom no coincideix.
    """
    m = _RE_CROP.match(path.stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def parse_char_filename(path: Path) -> tuple[str | None, int | None, int | None]:
    """
    Extreu (stem_base, box_idx, char_idx) de noms '{stem}_box{n}_char{i}.png'.
    Retorna (None, None, None) si no coincideix.
    """
    m = _RE_CHAR.match(path.name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None
