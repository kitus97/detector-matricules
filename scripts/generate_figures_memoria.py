"""
scripts/generate_figures_memoria.py
====================================
Genera les figures QUALITATIVES de la memòria tècnica (memoria/figures/),
executant el pipeline real (alpr/) sobre imatges d'exemple i desant-ne
totes les etapes intermèdies:

  1. pipeline_quali.png    — exemple end-to-end complet (memòria §3.1):
       imatge amb candidats → crop → binari amb bboxes → caràcters 28×28
       amb la predicció de la CNN i la lectura final.
  2. morfologia_closing.png — efecte del closing horitzontal (memòria §4.1):
       Sobel-v binaritzat → després de closing+opening → candidats filtrats.
  3. exemples_e2e.png      — els 3 exemples qualitatius de la taula 5.5:
       crop real + GT + lectura del sistema.

Ús
--
    python scripts/generate_figures_memoria.py
    python scripts/generate_figures_memoria.py --out-dir memoria/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARREL = Path(__file__).resolve().parents[1]
if str(ARREL) not in sys.path:
    sys.path.insert(0, str(ARREL))

from alpr import config
from alpr.common.io import load_image
from alpr.common.annotations import load_annotations
from alpr.common.geometry import iou
from alpr.detector.detector import detect_debug
from alpr.segmenter.segmenter import segmenta_caixa
from alpr.ocr.infer import load_model, predict
from alpr.reader.reader import read_plate

# Imatges d'exemple (les mateixes de la taula 5.5 de la memòria)
STEMS_EXEMPLES = ["eu1", "eu10", "eu11"]
STEM_PRINCIPAL = "eu1"          # exemple perfecte (M5XSX) per a la figura gran


# ══════════════════════════════════════════════════════════════════════════════
# Execució del pipeline amb captura d'etapes
# ══════════════════════════════════════════════════════════════════════════════

def _troba_imatge(stem: str) -> Path:
    """Retorna la ruta de la imatge {stem} a data/raw, sigui quina sigui l'extensió."""
    for ext in config.IMAGE_EXTENSIONS:
        p = config.DATA_RAW_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No s'ha trobat {stem}.* a {config.DATA_RAW_DIR}")


def _caixa_del_gt(boxes: list, gt_box: tuple) -> tuple | None:
    """La caixa candidata que conté el centre del GT amb més IoU (com a E9)."""
    gx, gy, gw, gh = gt_box
    cx, cy = gx + gw / 2.0, gy + gh / 2.0
    containing = [b for b in boxes
                  if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]]
    if not containing:
        return None
    return max(containing, key=lambda b: iou(b, gt_box))


def executa_pipeline(stem: str, model, idx_to_class, device) -> dict:
    """
    Executa el pipeline complet sobre la imatge {stem} i retorna TOTES les
    etapes intermèdies (detector debug + segmentador debug + prediccions OCR).
    """
    img = load_image(_troba_imatge(stem))
    anns = load_annotations(config.DATA_RAW_DIR / f"{stem}.txt")
    if not anns:
        raise ValueError(f"Sense GT per a {stem}")
    a = anns[0]
    gt_box, gt_plate = (a["x"], a["y"], a["w"], a["h"]), a["plate"]

    det = detect_debug(img)
    box = _caixa_del_gt(det["boxes"], gt_box)
    if box is None:
        raise ValueError(f"El detector no troba la matrícula de {stem}")

    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    seg = segmenta_caixa(crop)

    preds = [predict(c, model, idx_to_class, device) for c in seg["chars"]]
    lectura = read_plate([ch for ch, _ in preds if ch != "?"])

    return {
        "stem": stem, "img": img, "gt_plate": gt_plate, "gt_box": gt_box,
        "boxes": det["boxes"], "box": box, "crop": crop,
        "binary": det["binary"], "morph": det["morph"],
        "seg": seg, "preds": preds, "lectura": lectura,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def figura_pipeline(r: dict, out: Path) -> None:
    """Figura 1 — el pipeline complet sobre una imatge real (2 files)."""
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0], hspace=0.32, wspace=0.12)

    # (a) imatge original amb tots els candidats; el de la matrícula, destacat
    ax = fig.add_subplot(gs[0, 0])
    vis = r["img"].copy()
    for b in r["boxes"]:
        bx, by, bw, bh = b
        color = (0, 200, 50) if b == r["box"] else (60, 60, 230)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color,
                      3 if b == r["box"] else 2)
    ax.imshow(_bgr2rgb(vis))
    ax.set_title(f"(a) Detector: {len(r['boxes'])} candidats", fontsize=10)
    ax.axis("off")

    # (b) crop del candidat de la matrícula (alineat pel deskew)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_bgr2rgb(r["seg"]["aligned"]))
    ax.set_title(f"(b) Crop normalitzat i alineat ({r['seg']['angle']:+.1f}°)",
                 fontsize=10)
    ax.axis("off")

    # (c) binarització adaptativa amb els bboxes dels caràcters acceptats
    ax = fig.add_subplot(gs[0, 2])
    thr = cv2.cvtColor(r["seg"]["thresh"], cv2.COLOR_GRAY2RGB)
    for (bx, by, bw, bh) in r["seg"]["filtered_bboxes"]:
        cv2.rectangle(thr, (bx, by), (bx + bw, by + bh), (0, 220, 60), 2)
    ax.imshow(thr)
    ax.set_title(f"(c) Binarització + {len(r['seg']['filtered_bboxes'])} caràcters",
                 fontsize=10)
    ax.axis("off")

    # (d) fila inferior: els caràcters 28×28 amb la predicció de la CNN
    n = len(r["seg"]["chars"])
    gs_chars = gs[1, :].subgridspec(1, max(n, 1), wspace=0.15)
    for i, (char_img, (lletra, conf)) in enumerate(zip(r["seg"]["chars"], r["preds"])):
        ax = fig.add_subplot(gs_chars[0, i])
        ax.imshow(char_img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{lletra}\n{conf:.0%}", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Pipeline complet sobre {r['stem']}:  GT = {r['gt_plate']}   →   "
        f"lectura = {r['lectura']}",
        fontsize=12, y=0.99,
    )
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def figura_morfologia(r: dict, out: Path) -> None:
    """Figura 2 — l'efecte del closing horitzontal (Fase 1, pas 3)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    axes[0].imshow(r["binary"], cmap="gray")
    axes[0].set_title("(a) Sobel vertical binaritzat (Otsu)", fontsize=10)

    axes[1].imshow(r["morph"], cmap="gray")
    axes[1].set_title("(b) Després del closing $15\\times3$ + opening", fontsize=10)

    vis = r["img"].copy()
    for (bx, by, bw, bh) in r["boxes"]:
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 200, 50), 2)
    axes[2].imshow(_bgr2rgb(vis))
    axes[2].set_title("(c) Components que passen el filtre de forma", fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def figura_exemples(resultats: list[dict], out: Path) -> None:
    """Figura 3 — els crops reals dels exemples de la taula end-to-end."""
    fig, axes = plt.subplots(len(resultats), 1, figsize=(7, 1.55 * len(resultats)))
    if len(resultats) == 1:
        axes = [axes]
    for ax, r in zip(axes, resultats):
        ax.imshow(_bgr2rgb(r["seg"]["aligned"]))
        marca = "✓" if r["lectura"] == r["gt_plate"] else "≈"
        ax.set_title(f"{r['stem']}:  GT = {r['gt_plate']}   →   "
                     f"lectura = {r['lectura']}  {marca}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=ARREL / "memoria" / "figures",
                        help="On desar les figures (default: memoria/figures)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, _c2i, idx_to_class, device = load_model(None)

    print("Executant el pipeline sobre els exemples…")
    resultats = [executa_pipeline(s, model, idx_to_class, device)
                 for s in STEMS_EXEMPLES]
    principal = next(r for r in resultats if r["stem"] == STEM_PRINCIPAL)

    print(f"Generant figures a {args.out_dir} …")
    figura_pipeline(principal, args.out_dir / "pipeline_quali.png")
    figura_morfologia(principal, args.out_dir / "morfologia_closing.png")
    figura_exemples(resultats, args.out_dir / "exemples_e2e.png")

    for r in resultats:
        print(f"  {r['stem']}: GT={r['gt_plate']}  lectura={r['lectura']}  "
              f"({len(r['seg']['chars'])} chars, angle {r['seg']['angle']:+.1f}°)")


if __name__ == "__main__":
    main()
