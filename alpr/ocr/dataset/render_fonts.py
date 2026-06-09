"""
alpr/ocr/dataset/render_fonts.py
=================================
Fase 3 — prep dades: renderitza caràcters alfanumèrics des de fonts .ttf/.otf.

Genera imatges base 64×64 px, grisos, caràcter NEGRE sobre fons BLANC.
No aplica cap augmentation: les imatges base han de ser netes.
L'augmentation es fa a augment.py.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from alpr import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Funcions
# ══════════════════════════════════════════════════════════════════════════════

def find_fonts(fonts_dir: Path) -> list[Path]:
    """
    Retorna la llista de tots els .ttf i .otf a fonts_dir (recursiu).
    Llança FileNotFoundError si no en troba cap.
    """
    fonts = sorted(fonts_dir.rglob("*.ttf")) + sorted(fonts_dir.rglob("*.otf"))
    if not fonts:
        raise FileNotFoundError(
            f"No s'han trobat fonts .ttf/.otf a '{fonts_dir}'."
        )
    return fonts


def fit_font(
    font_path: Path, char: str, target_size: int
) -> ImageFont.FreeTypeFont:
    """
    Tria la mida de font (en punts) perquè el caràcter encaixi dins
    (target_size - 2*MARGIN) × (target_size - 2*MARGIN) px.
    """
    max_dim   = target_size - 2 * config.MARGIN
    best_font = ImageFont.truetype(str(font_path), size=8)

    for size in range(8, 200):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = font.getbbox(char)
        if (bbox[2] - bbox[0]) > max_dim or (bbox[3] - bbox[1]) > max_dim:
            break
        best_font = font

    return best_font


def render_char(
    char: str, font: ImageFont.FreeTypeFont, canvas_size: int
) -> Image.Image:
    """
    Renderitza un únic caràcter centrat en un llenç blanc canvas_size×canvas_size.
    Color: fons blanc (255), caràcter negre (0).
    """
    img  = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    cw   = bbox[2] - bbox[0]
    ch   = bbox[3] - bbox[1]
    x    = (canvas_size - cw) // 2 - bbox[0]
    y    = (canvas_size - ch) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return img


def render_all(
    fonts_dir: Path,
    out_dir: Path,
    canvas_size: int | None = None,
) -> None:
    """
    Pipeline complet: per cada font i caràcter, renderitza i guarda.

    Estructura de sortida:
        out_dir/{clase}/{font_stem}_{clase}.png
    """
    if canvas_size is None:
        canvas_size = config.CANVAS_SIZE

    fonts = find_fonts(fonts_dir)
    chars = config.CHARS

    log.info(f"Fonts trobades     : {len(fonts)}")
    log.info(f"Caràcters per font : {len(chars)}")
    log.info(f"Total imatges base : {len(fonts) * len(chars)}")

    for char in chars:
        (out_dir / char).mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for font_path in fonts:
        for char in chars:
            out_path = out_dir / char / f"{font_path.stem}_{char}.png"
            if out_path.exists():
                n_skip += 1
                continue
            try:
                font = fit_font(font_path, char, canvas_size)
                img  = render_char(char, font, canvas_size)
                img.save(str(out_path))
                n_ok += 1
            except Exception as e:
                log.warning(f"  ERROR {font_path.stem}/'{char}': {e}")
                n_skip += 1
        log.info(f"  OK  {font_path.stem}")

    log.info(f"Imatges generades : {n_ok}  |  Saltades/error : {n_skip}")
