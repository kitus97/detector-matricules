"""
alpr/ocr/eval_real.py
=====================
Fase 3 — avaluació del model CharCNN sobre caràcters REALS segmentats.

Opera sobre els PNGs que el segmentador deixa a `data/chars/`
(`{stem}_box{n}_char{i}.png`), els classifica amb la CNN, filtra els de baixa
confiança i compara la seqüència resultant amb el ground truth de `data/raw/`
via distància d'edició (Levenshtein) — robusta a desalineacions de longitud.

Genera: resum de mètriques (char/plate accuracy ED, posicional, gap sintètic→real),
`models/eval_real_results.csv` i `models/confusion_matrix_real.png`.

Reutilitza `load_model`, `predict_char` i `filter_by_confidence` d'`alpr.ocr.infer`,
i el parseig d'`alpr.common.annotations` (cap lògica duplicada).

Ús
--
    python -m alpr.ocr.eval_real
    python -m alpr.ocr.eval_real --conf-threshold 0.6
    python -m alpr.ocr.eval_real --strict-length
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from alpr import config
from alpr.common.annotations import load_annotations, parse_char_filename
from alpr.ocr.infer import load_model, predict_char, filter_by_confidence

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth i distància d'edició
# ══════════════════════════════════════════════════════════════════════════════

def parse_gt(gt_dir: Path, stem: str) -> list[str]:
    """Retorna les matrícules GT de `{gt_dir}/{stem}.txt` (majúscules)."""
    return [a["plate"] for a in load_annotations(gt_dir / f"{stem}.txt")]


def edit_distance(s1: str, s2: str) -> int:
    """
    Distància d'edició (inserció/eliminació/substitució) entre dues cadenes.
    Programació dinàmica amb O(min(m, n)) d'espai.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * n
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


# ══════════════════════════════════════════════════════════════════════════════
# Agrupació de caràcters per ROI
# ══════════════════════════════════════════════════════════════════════════════

def group_files(chars_dir: Path, gt_dir: Path) -> tuple[list[dict], int]:
    """
    Agrupa els PNGs per `{stem}_box{n}` i associa el GT del stem. No comprova
    longitud (tots els grups amb GT es retornen). Retorna (groups, n_sense_gt).
    """
    raw: dict[str, list] = defaultdict(list)
    for fpath in sorted(chars_dir.glob("*.png")):
        stem, box_n, char_i = parse_char_filename(fpath)
        if stem is None:
            log.warning(f"Nom no reconegut, ignorat: {fpath.name}")
            continue
        raw[f"{stem}_box{box_n}"].append((char_i, fpath, stem, box_n))

    groups: list[dict] = []
    n_sense_gt = 0
    for group_key, entries in sorted(raw.items()):
        entries.sort(key=lambda e: e[0])
        stem, box_n = entries[0][2], entries[0][3]
        plates_gt = parse_gt(gt_dir, stem)
        if not plates_gt:
            n_sense_gt += 1
            continue
        groups.append({
            "group_key": group_key,
            "stem":      stem,
            "box":       box_n,
            "entries":   [(e[0], e[1]) for e in entries],
            "plates_gt": plates_gt,
        })
    return groups, n_sense_gt


# ══════════════════════════════════════════════════════════════════════════════
# Avaluació d'un grup
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_group(
    model,
    group: dict,
    threshold: float,
    idx_to_class: dict[int, str],
    device,
    strict_length: bool,
) -> dict | None:
    """Inferència → filtratge per confiança → comparació amb el millor GT (ED)."""
    all_preds: list[dict] = []
    for char_idx, img_path in group["entries"]:
        pred_idx, conf = predict_char(model, img_path, device)
        all_preds.append({
            "char_idx":     char_idx,
            "pred":         idx_to_class.get(pred_idx, "?"),
            "confidence":   conf,
            "filtered_out": False,
        })

    retained, discarded = filter_by_confidence(all_preds, threshold)
    for p in discarded:
        p["filtered_out"] = True

    pred_str = "".join(p["pred"] for p in retained)
    best_gt  = min(group["plates_gt"], key=lambda g: edit_distance(pred_str, g))
    gt_len   = len(best_gt)
    ed       = edit_distance(pred_str, best_gt)

    if strict_length and len(retained) != gt_len:
        return None

    positional_pairs: list[dict] = []
    if len(retained) == gt_len:
        for pred_info, gt_char in zip(retained, best_gt):
            positional_pairs.append({
                "gt":      gt_char,
                "pred":    pred_info["pred"],
                "correct": gt_char == pred_info["pred"],
            })

    return {
        "group_key":        group["group_key"],
        "stem":             group["stem"],
        "box":              group["box"],
        "gt":               best_gt,
        "pred_str":         pred_str,
        "n_segmented":      len(all_preds),
        "n_after_filter":   len(retained),
        "n_discarded":      len(discarded),
        "gt_length":        gt_len,
        "edit_distance":    ed,
        "plate_match":      ed == 0,
        "lengths_match":    len(retained) == gt_len,
        "positional_pairs": positional_pairs,
        "all_preds":        all_preds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mètriques agregades
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    """Agrega char/plate accuracy (ED) i accuracy posicional sobre els grups."""
    if not results:
        return {}

    n_groups     = len(results)
    total_ed     = sum(r["edit_distance"] for r in results)
    total_gt_len = sum(r["gt_length"]     for r in results)
    n_match      = sum(r["plate_match"]   for r in results)

    pos_pairs = [p for r in results for p in r["positional_pairs"]]
    char_acc_pos = (
        sum(p["correct"] for p in pos_pairs) / len(pos_pairs) if pos_pairs else None
    )

    return {
        "n_groups":     n_groups,
        "total_ed":     total_ed,
        "total_gt_len": total_gt_len,
        "n_plate_match": n_match,
        "char_acc_ed":  max(0.0, 1.0 - total_ed / total_gt_len) if total_gt_len else 0.0,
        "plate_acc_ed": n_match / n_groups,
        "char_acc_pos": char_acc_pos,
        "n_pos_groups": sum(1 for r in results if r["lengths_match"]),
        "n_pos_pairs":  len(pos_pairs),
    }


def plot_confusion(y_true, y_pred, classes, out_path: Path) -> list[tuple[str, str, int]]:
    """Guarda la matriu de confusió (normalitzada per fila) i retorna el top-10 fora diagonal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n  = len(classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fracció per fila (classe real)")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=6, rotation=90)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Caràcter predit"); ax.set_ylabel("Caràcter real")
    ax.set_title("Matriu de confusió — caràcters reals (grups alineats, normalitzada per fila)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    log.info(f"Matriu de confusió guardada a: {out_path}")

    off = [(classes[i], classes[j], int(cm[i, j]))
           for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    off.sort(key=lambda x: -x[2])
    return off[:10]


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(results: list[dict], csv_path: Path) -> None:
    """Desa un CSV amb una fila per caràcter (inclou els filtrats per confiança)."""
    fieldnames = ["stem", "box", "char_idx", "gt", "pred", "confidence", "correct",
                  "filtered_out", "n_segmented", "n_after_filter", "gt_length",
                  "edit_distance", "n_discarded_lowconf"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            retained = [p for p in r["all_preds"] if not p["filtered_out"]]
            gt_map: dict[int, str] = {}
            if r["lengths_match"]:
                for pred_info, gt_char in zip(retained, r["gt"]):
                    gt_map[id(pred_info)] = gt_char
            for p in r["all_preds"]:
                if r["lengths_match"] and not p["filtered_out"]:
                    gt_char = gt_map.get(id(p), "?")
                    correct = gt_char == p["pred"]
                else:
                    gt_char, correct = "?", False
                writer.writerow({
                    "stem": r["stem"], "box": r["box"], "char_idx": p["char_idx"],
                    "gt": gt_char, "pred": p["pred"],
                    "confidence": f"{p['confidence']:.4f}", "correct": correct,
                    "filtered_out": p["filtered_out"],
                    "n_segmented": r["n_segmented"], "n_after_filter": r["n_after_filter"],
                    "gt_length": r["gt_length"], "edit_distance": r["edit_distance"],
                    "n_discarded_lowconf": r["n_discarded"],
                })
    log.info(f"CSV guardat a: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Avalua CharCNN sobre caràcters reals.")
    parser.add_argument("--chars-dir", type=Path, default=config.DATA_CHARS_DIR,
                        help=f"Directori amb els PNGs segmentats (default: {config.DATA_CHARS_DIR})")
    parser.add_argument("--gt-dir", type=Path, default=config.DATA_RAW_DIR,
                        help=f"Directori amb els .txt de GT (default: {config.DATA_RAW_DIR})")
    parser.add_argument("--model", type=Path, default=None,
                        help=f"Checkpoint (default: {config.MODEL_CNN_PATH})")
    parser.add_argument("--out-dir", type=Path, default=config.MODELS_DIR,
                        help=f"On desar CSV i figura (default: {config.MODELS_DIR})")
    parser.add_argument("--conf-threshold", type=float, default=config.CONF_THRESHOLD,
                        help=f"Llindar de confiança (default: {config.CONF_THRESHOLD})")
    parser.add_argument("--strict-length", action="store_true",
                        help="Descarta grups on la longitud filtrada ≠ longitud GT")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups, n_sense_gt = group_files(args.chars_dir, args.gt_dir)
    if not groups:
        log.error(f"Cap grup amb GT a '{args.chars_dir}'. Executa primer el segmentador.")
        return
    log.info(f"Grups amb GT: {len(groups)}  (sense GT: {n_sense_gt})")

    model, class_to_idx, idx_to_class, device = load_model(args.model)

    results: list[dict] = []
    n_strict = 0
    for g in groups:
        r = evaluate_group(model, g, args.conf_threshold, idx_to_class, device,
                           args.strict_length)
        if r is None:
            n_strict += 1
        else:
            results.append(r)
    if not results:
        log.error("Tots els grups descartats (--strict-length).")
        return

    metrics = compute_metrics(results)

    classes_sorted = sorted(class_to_idx, key=lambda c: class_to_idx[c])
    pos_pairs = [p for r in results for p in r["positional_pairs"]]
    top10: list = []
    if pos_pairs:
        top10 = plot_confusion([p["gt"] for p in pos_pairs], [p["pred"] for p in pos_pairs],
                               classes_sorted, args.out_dir / "confusion_matrix_real.png")

    write_csv(results, args.out_dir / "eval_real_results.csv")

    # ─── Resum ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AVALUACIÓ SOBRE CARÀCTERS REALS")
    print("=" * 60)
    print(f"  Grups avaluats          : {metrics['n_groups']}"
          + (f"  (descartats strict: {n_strict})" if n_strict else ""))
    print(f"  Char Accuracy (ED)      : {metrics['char_acc_ed']:.2%}"
          f"  (ED={metrics['total_ed']}, chars GT={metrics['total_gt_len']})")
    print(f"  Plate Accuracy (ED=0)   : {metrics['plate_acc_ed']:.2%}"
          f"  ({metrics['n_plate_match']}/{metrics['n_groups']} perfectes)")
    if metrics["char_acc_pos"] is not None:
        print(f"  Char Accuracy posicional: {metrics['char_acc_pos']:.2%}"
              f"  ({metrics['n_pos_pairs']} caràcters, {metrics['n_pos_groups']} grups alineats)")
    if top10:
        print("\n  Top parells de confusió (real → predit):")
        for gt_c, pred_c, cnt in top10:
            print(f"    {gt_c} → {pred_c}  ({cnt})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
