"""
01_render_fonts.py
==================
Pas 2 del pla d'entrenament OCR.

Recorre totes les fonts .ttf / .otf de resources/fonts/, renderitza cada un
dels 62 caràcters alfanumèrics (A-Z, 0-9) i guarda les imatges base a:

    data/synthetic/raw/{classe}/{font_stem}_{classe}.png

Cada imatge és en escala de grisos, 64×64 px, negre sobre blanc.
No s'aplica cap augmentation aquí — les imatges base han de ser netes.
L'augmentation es fa al pas següent (02_augment.py).

Ús
--
    python 01_render_fonts.py
    python 01_render_fonts.py --fonts_dir resources/fonts --out_dir data/synthetic/raw --size 64
"""

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ─── Constants ────────────────────────────────────────────────────────────────

# Tots els caràcters que pot contenir una matrícula
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # 62 classes

# Mida del llenç de sortida (en píxels)
CANVAS_SIZE = 64

# Marge interior: el caràcter ocupa com a màxim CANVAS_SIZE - 2*MARGIN px
MARGIN = 6

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Funcions principals
# ══════════════════════════════════════════════════════════════════════════════

def find_fonts(fonts_dir: Path) -> list[Path]:
    """
    Retorna la llista de tots els fitxers .ttf i .otf que hi ha a fonts_dir,
    recursivament (per si les fonts estan en subcarpetes).
    """
    ttf = list(fonts_dir.rglob("*.ttf"))
    otf = list(fonts_dir.rglob("*.otf"))
    fonts = sorted(ttf + otf)
    if not fonts:
        raise FileNotFoundError(
            f"No s'han trobat fonts .ttf/.otf a '{fonts_dir}'. "
            "Comprova que el directori és correcte."
        )
    return fonts


def fit_font(font_path: Path, char: str, target_size: int) -> ImageFont.FreeTypeFont:
    """
    Tria la mida de font (en punts) perquè el caràcter encaixi dins
    target_size × target_size px amb un marge MARGIN a cada costat.

    La cerca és binària: prova mides creixents fins que el caràcter supera
    l'àrea disponible, i retorna la mida anterior.

    Retorna un objecte ImageFont llest per usar amb ImageDraw.
    """
    max_dim = target_size - 2 * MARGIN  # amplada i alçada màximes del caràcter
    font_size = 8
    best_font = ImageFont.truetype(str(font_path), size=font_size)

    for font_size in range(8, 200):
        font = ImageFont.truetype(str(font_path), size=font_size)
        # getbbox retorna (left, top, right, bottom) del caràcter
        bbox = font.getbbox(char)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        if char_w > max_dim or char_h > max_dim:
            break
        best_font = font

    return best_font


def render_char(char: str, font: ImageFont.FreeTypeFont,
                canvas_size: int) -> Image.Image:
    """
    Renderitza un únic caràcter centrat en un llenç blanc de canvas_size×canvas_size.

    Convenció de color:
      - Fons  : blanc  (255)
      - Caràcter: negre (0)

    El segmentador lliura imatges binaritzades amb caràcters BLANCS sobre fons NEGRE.
    Aquí ho invertim deliberadament: treballem en l'espai natural (tinta fosca sobre
    paper blanc) i deixem la inversió per al pas d'augmentation + binarització,
    on s'aplicarà el mateix adaptiveThreshold que el segmentador real.
    """
    img = Image.new("L", (canvas_size, canvas_size), color=255)  # fons blanc
    draw = ImageDraw.Draw(img)

    # Calculem la posició centrada
    bbox = font.getbbox(char)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]
    x = (canvas_size - char_w) // 2 - bbox[0]
    y = (canvas_size - char_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=0)  # text negre
    return img


def render_all(fonts_dir: Path, out_dir: Path, canvas_size: int) -> None:
    """
    Pipeline complet: per cada font i per cada caràcter, renderitza i guarda.

    Estructura de sortida:
        out_dir/
            A/  font1_A.png  font2_A.png ...
            B/  ...
            0/  ...
    """
    fonts = find_fonts(fonts_dir)
    log.info(f"Fonts trobades     : {len(fonts)}")
    log.info(f"Caràcters per font : {len(CHARS)}")
    log.info(f"Total imatges base : {len(fonts) * len(CHARS)}")
    log.info(f"Directori sortida  : {out_dir.resolve()}")
    log.info("-" * 50)

    # Crea les carpetes de classe si no existeixen
    for char in CHARS:
        # Els dígits i lletres van directament com a nom de carpeta.
        # No hi ha conflicte en cap SO perquè cap nom és reservat.
        (out_dir / char).mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0

    for font_path in fonts:
        font_stem = font_path.stem  # nom del fitxer sense extensió

        for char in CHARS:
            out_path = out_dir / char / f"{font_stem}_{char}.png"

            # Salta si ja existeix (permet reprendre una execució interrompuda)
            if out_path.exists():
                n_skip += 1
                continue

            try:
                font = fit_font(font_path, char, canvas_size)
                img = render_char(char, font, canvas_size)
                img.save(str(out_path))
                n_ok += 1
            except Exception as e:
                log.warning(f"  ERROR  {font_stem} / '{char}': {e}")
                n_skip += 1

        log.info(f"  OK  {font_stem}")

    log.info("-" * 50)
    log.info(f"Imatges generades : {n_ok}")
    log.info(f"Saltades / error  : {n_skip}")
    log.info(f"Resultat a        : {out_dir.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# Verificació visual (opcional)
# ══════════════════════════════════════════════════════════════════════════════

def show_sample(out_dir: Path, n_chars: int = 10, n_fonts: int = 3) -> None:
    """
    Mostra una graella de mostra: n_chars columnes × n_fonts files.
    Útil per verificar visualment que el renderitzat és correcte.
    Requereix matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import random
    except ImportError:
        log.warning("matplotlib no disponible. Salta la verificació visual.")
        return

    sample_chars = CHARS[:n_chars]
    fig, axes = plt.subplots(n_fonts, n_chars, figsize=(n_chars * 1.2, n_fonts * 1.5))

    for row, char in enumerate(sample_chars):
        char_dir = out_dir / char
        imgs = sorted(char_dir.glob("*.png"))
        if not imgs:
            continue
        sample_fonts = random.sample(imgs, min(n_fonts, len(imgs)))
        for col, img_path in enumerate(sample_fonts):
            ax = axes[col][row] if n_fonts > 1 else axes[row]
            ax.imshow(Image.open(img_path), cmap="gray", vmin=0, vmax=255)
            ax.set_title(char, fontsize=8)
            ax.axis("off")

    plt.suptitle("Mostra d'imatges base renderitzades", fontsize=11)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderitza caràcters alfanumèrics des de fonts .ttf/.otf."
    )
    parser.add_argument(
        "--fonts_dir", default="resources/fonts",
        help="Directori amb les fonts .ttf/.otf (default: resources/fonts)"
    )
    parser.add_argument(
        "--out_dir", default="data/synthetic/raw",
        help="Directori de sortida (default: data/synthetic/raw)"
    )
    parser.add_argument(
        "--size", type=int, default=CANVAS_SIZE,
        help=f"Mida del llenç en píxels (default: {CANVAS_SIZE})"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostra una graella visual de mostra al final"
    )
    args = parser.parse_args()

    render_all(
        fonts_dir=Path(args.fonts_dir),
        out_dir=Path(args.out_dir),
        canvas_size=args.size,
    )

    if args.preview:
        show_sample(Path(args.out_dir))


if __name__ == "__main__":
    main()
