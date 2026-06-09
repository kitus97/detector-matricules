"""
main.py
=======
Pipeline end-to-end de detecció i lectura de matrícules (ALPR).

Flux complet per imatge:
  imatge BGR
    └─ Fase 1 · detector    → retalls candidats (màxim recall)
    └─ Fase 2 · segmenter   → caràcters 28×28 per cada retall vàlid
    └─ Fase 3 · ocr/infer   → lletra + confiança per cada caràcter
    └─ Fase 4 · reader      → matrícula final (correcció O↔0, I↔1…)

Ús
--
    # Una sola imatge
    python main.py --input data/raw/foto.jpg

    # Tot un directori
    python main.py --input data/raw/

    # Desar resultats en CSV
    python main.py --input data/raw/ --output output/results.csv

    # Mode debug (mostra cada etapa per pantalla)
    python main.py --input data/raw/foto.jpg --debug
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from alpr import config
from alpr.common.io import load_image, iter_images
from alpr.detector.detector import detect_boxes, detect_debug
from alpr.segmenter.segmenter import segment, segmenta_caixa
from alpr.ocr.infer import load_model, predict
from alpr.reader.reader import read_plate

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline per a una sola imatge
# ══════════════════════════════════════════════════════════════════════════════

def process_image(
    img_bgr: np.ndarray,
    model,
    idx_to_class: dict,
    device,
    debug: bool = False,
) -> list[dict]:
    """
    Executa el pipeline complet sobre una imatge BGR.

    Retorna llista de dicts amb:
      { 'box': (x,y,w,h), 'plate': str, 'n_chars': int,
        'chars': list[str], 'confs': list[float], 'mean_conf': float }

    Llista buida si no es detecta cap matrícula vàlida.
    """
    results = []

    # ── Fase 1: Detecció ────────────────────────────────────────────────────
    if debug:
        det = detect_debug(img_bgr)
        boxes = det["boxes"]
        log.debug(f"  Detector: {det['n_total']} components → {len(boxes)} candidats")
    else:
        boxes = detect_boxes(img_bgr)

    if not boxes:
        return results

    # ── Fases 2–4: per cada candidat ─────────────────────────────────────────
    for box in boxes:
        x, y, w, h = box
        roi = img_bgr[y : y + h, x : x + w]

        if debug:
            seg = segmenta_caixa(roi)
            chars_imgs = seg["chars"]
            accepted = seg["accepted"]
            if not accepted:
                log.debug(f"  ROI ({x},{y},{w},{h}) rebutjada: {seg['rejection_reason']}")
                continue
        else:
            chars_imgs = segment(roi)
            if not chars_imgs:
                continue

        # Fase 3 — OCR per caràcter
        char_preds = [
            predict(c, model, idx_to_class, device)
            for c in chars_imgs
        ]
        chars  = [p[0] for p in char_preds]
        confs  = [p[1] for p in char_preds]

        # Filtra '?' (baixa confiança)
        valid = [(ch, cf) for ch, cf in zip(chars, confs) if ch != "?"]
        if not valid:
            continue
        chars_v = [v[0] for v in valid]
        confs_v = [v[1] for v in valid]

        # Fase 4 — Reader (correcció contextual)
        plate     = read_plate(chars_v)
        mean_conf = float(np.mean(confs_v))

        results.append({
            "box":       box,
            "plate":     plate,
            "n_chars":   len(chars_v),
            "chars":     chars_v,
            "confs":     confs_v,
            "mean_conf": mean_conf,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualització debug
# ══════════════════════════════════════════════════════════════════════════════

def draw_results(
    img_bgr: np.ndarray,
    results: list[dict],
) -> np.ndarray:
    """Dibuixa bboxes i text de matrícula sobre la imatge. Retorna còpia."""
    out = img_bgr.copy()
    for r in results:
        x, y, w, h = r["box"]
        plate      = r["plate"]
        conf       = r["mean_conf"]

        color = (0, 200, 50)  # verd
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label = f"{plate} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Bucle principal
# ══════════════════════════════════════════════════════════════════════════════

def run(
    input_path: Path,
    output_csv: Path | None,
    model_path: Path | None,
    debug: bool,
    show: bool,
) -> None:
    """
    Executa el pipeline sobre input_path (imatge o directori).
    Opcionalment guarda els resultats en output_csv.
    """
    # ── Càrrega del model OCR ───────────────────────────────────────────────
    log.info("Carregant model OCR…")
    try:
        model, _class_to_idx, idx_to_class, device = load_model(model_path)
    except FileNotFoundError as e:
        log.error(str(e))
        log.error(
            "Entrena primer el model amb:\n"
            "  python scripts/train_ocr.py"
        )
        sys.exit(1)

    # ── Llista d'imatges a processar ─────────────────────────────────────────
    if input_path.is_file():
        images = [(input_path, load_image(input_path))]
    elif input_path.is_dir():
        images = list(iter_images(input_path))
    else:
        log.error(f"Ruta no vàlida: '{input_path}'")
        sys.exit(1)

    if not images:
        log.warning(f"No s'han trobat imatges a '{input_path}'")
        return

    log.info(f"Processant {len(images)} imatge(s)…")
    log.info("-" * 60)

    # ── Preparació CSV ──────────────────────────────────────────────────────
    csv_rows: list[dict] = []

    # ── Bucle d'imatges ──────────────────────────────────────────────────────
    t0_global = time.time()

    for img_path, img_bgr in images:
        t0 = time.time()
        results = process_image(img_bgr, model, idx_to_class, device, debug=debug)
        elapsed = time.time() - t0

        if results:
            for r in results:
                plate = r["plate"]
                conf  = r["mean_conf"]
                log.info(f"  {img_path.name:<30}  {plate:<10}  conf={conf:.0%}  ({elapsed*1000:.0f} ms)")
                csv_rows.append({
                    "file":      img_path.name,
                    "plate":     plate,
                    "n_chars":   r["n_chars"],
                    "mean_conf": f"{conf:.4f}",
                    "box":       str(r["box"]),
                })
        else:
            log.info(f"  {img_path.name:<30}  [sense detecció]  ({elapsed*1000:.0f} ms)")
            csv_rows.append({
                "file":      img_path.name,
                "plate":     "",
                "n_chars":   0,
                "mean_conf": "",
                "box":       "",
            })

        # Visualització opcional
        if show or debug:
            annotated = draw_results(img_bgr, results)
            cv2.imshow(img_path.name, annotated)
            key = cv2.waitKey(0 if debug else 500)
            cv2.destroyAllWindows()
            if key == ord("q"):
                log.info("Interromput per l'usuari.")
                break

    total = time.time() - t0_global
    n_detected = sum(1 for r in csv_rows if r["plate"])
    log.info("-" * 60)
    log.info(f"Resultat: {n_detected}/{len(images)} imatges amb matrícula detectada")
    log.info(f"Temps total: {total:.1f} s  ({total/len(images)*1000:.0f} ms/imatge)")

    # ── Desa CSV ─────────────────────────────────────────────────────────────
    if output_csv and csv_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "plate", "n_chars", "mean_conf", "box"]
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        log.info(f"Resultats desats a: {output_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline ALPR: detecció + segmentació + OCR de matrícules.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Imatge .jpg/.png o directori amb imatges.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Fitxer CSV de sortida (opcional). Ex: output/results.csv",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Ruta al checkpoint OCR (default: {config.MODEL_CNN_PATH})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mode debug: mostra etapes intermèdies i obre finestres OpenCV.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Mostra el resultat visual de cada imatge.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Activa logs de nivell DEBUG.",
    )
    args = parser.parse_args()

    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(
        input_path = Path(args.input),
        output_csv = Path(args.output) if args.output else None,
        model_path = Path(args.model) if args.model else None,
        debug      = args.debug,
        show       = args.show,
    )


if __name__ == "__main__":
    main()
