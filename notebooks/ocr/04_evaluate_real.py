"""
04_evaluate_real.py
===================
Pas 9 del pla d'entrenament OCR.

Avalua el model CharCNN sobre caràcters reals segmentats. En lloc de descartar
grups per desalineació de longitud, executa el CNN sobre TOTS els caràcters,
filtra els de baixa confiança (possibles falsos positius del segmentador) i
compara la seqüència resultant amb el GT via distància d'edició (Levenshtein).

Ús
--
    python 04_evaluate_real.py
    python 04_evaluate_real.py --conf_threshold 0.6 --synthetic_acc 0.981
    python 04_evaluate_real.py --strict_length   # comportament original
"""

import argparse
import csv
import importlib.util
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # cal cridar-lo abans d'importar pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms


# ─── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_FNAME_RE = re.compile(r"^(.+)_box(\d+)_char(\d+)\.png$")


# ══════════════════════════════════════════════════════════════════════════════
# Importació de CharCNN
# ══════════════════════════════════════════════════════════════════════════════

def _import_charcnn() -> type:
    """
    Importa CharCNN des de 03_train_cnn.py via importlib perquè el nom del
    mòdul comença amb un dígit (no es pot fer amb import directe).
    Garanteix que l'arquitectura és idèntica a la de l'entrenament.
    """
    train_script = Path(__file__).parent / "03_train_cnn.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '03_train_cnn.py' a '{train_script.parent}'.\n"
            "Assegura't que els dos scripts són al mateix directori."
        )
    spec = importlib.util.spec_from_file_location("_train_cnn_module", train_script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CharCNN


# ══════════════════════════════════════════════════════════════════════════════
# Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

def parse_gt(gt_dir: Path, stem: str) -> list[str]:
    """
    Llegeix '{gt_dir}/{stem}.txt' i retorna la llista de matrícules (majúscules).
    Format per línia: nom_imatge x y w h MATRICULA (tabuladors o espais).
    """
    gt_path = gt_dir / f"{stem}.txt"
    if not gt_path.exists():
        return []

    plates = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            log.warning(f"Línia GT mal formada a '{gt_path}': {line!r}")
            continue
        plates.append(parts[-1].upper())
    return plates


# ══════════════════════════════════════════════════════════════════════════════
# Distància d'edició (Levenshtein)
# ══════════════════════════════════════════════════════════════════════════════

def edit_distance(s1: str, s2: str) -> int:
    """
    Distància d'edició (inserció, eliminació, substitució) entre dues cadenes.
    Implementació DP amb O(min(m,n)) d'espai.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1   # s1 sempre és la més llarga
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
# Agrupació de fitxers (sense comprovació de longitud)
# ══════════════════════════════════════════════════════════════════════════════

def group_files(
    segmented_dir: Path,
    gt_dir: Path,
) -> tuple[list[dict], int]:
    """
    Agrupa els PNGs per '{stem}_box{n}' i associa el GT de cada stem.
    NO comprova la longitud: tots els grups amb GT vàlid es retornen.

    Retorna:
      groups    : llista de dicts (group_key, stem, box, entries, plates_gt)
      n_no_gt   : grups descartats per falta de GT
    """
    raw: dict[str, list] = defaultdict(list)
    for fpath in sorted(segmented_dir.glob("*.png")):
        m = _FNAME_RE.match(fpath.name)
        if not m:
            log.warning(f"Nom de fitxer no reconegut: '{fpath.name}' — ignorat")
            continue
        stem   = m.group(1)
        box_n  = int(m.group(2))
        char_i = int(m.group(3))
        raw[f"{stem}_box{box_n}"].append((char_i, fpath, stem, box_n))

    groups: list[dict] = []
    n_no_gt = 0

    for group_key, entries in sorted(raw.items()):
        entries.sort(key=lambda e: e[0])
        stem  = entries[0][2]
        box_n = entries[0][3]

        plates_gt = parse_gt(gt_dir, stem)
        if not plates_gt:
            log.warning(f"Grup '{group_key}': sense GT — descartat")
            n_no_gt += 1
            continue

        groups.append({
            "group_key": group_key,
            "stem":      stem,
            "box":       box_n,
            "entries":   [(e[0], e[1]) for e in entries],   # (char_idx, path)
            "plates_gt": plates_gt,
        })

    return groups, n_no_gt


# ══════════════════════════════════════════════════════════════════════════════
# Càrrega del model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_path: Path,
) -> tuple[nn.Module, dict[str, int], dict[int, str], torch.device, float | None]:
    """
    Carrega el checkpoint i reconstrueix CharCNN.
    Retorna (model, class_to_idx, idx_to_class, device, val_acc_sintetica).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No s'ha trobat el checkpoint a '{model_path}'.\n"
            "Executa primer 03_train_cnn.py per generar-lo."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    n_classes    = ckpt["n_classes"]
    class_to_idx = ckpt["class_to_idx"]
    ckpt_val_acc = ckpt.get("val_acc")

    CharCNN = _import_charcnn()
    model = CharCNN(n_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if ckpt_val_acc is not None:
        log.info(f"Model carregat: {n_classes} classes | val_acc sintètica = {ckpt_val_acc:.2%}")
    else:
        log.info(f"Model carregat: {n_classes} classes")

    return model, class_to_idx, idx_to_class, device, ckpt_val_acc


# ══════════════════════════════════════════════════════════════════════════════
# Inferència per caràcter
# ══════════════════════════════════════════════════════════════════════════════

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def predict_char(
    model: nn.Module,
    img_path: Path,
    device: torch.device,
) -> tuple[int, float]:
    """
    Fa el forward pass d'una imatge de caràcter.
    Retorna (idx_predit, confiança_softmax).
    """
    img    = Image.open(img_path).convert("L")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    return pred_idx, confidence


# ══════════════════════════════════════════════════════════════════════════════
# Filtratge per confiança
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_confidence(
    preds: list[dict],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """
    Divideix les prediccions en retingudes (conf >= threshold) i descartades.
    Preserva l'ordre per char_idx.

    RISC D'ALINEACIÓ: si un caràcter REAL té confiança baixa (soroll extrem),
    es descarta incorrectament i la seqüència resultant es desalinea. Per
    aquest motiu la distància d'edició és la mètrica principal, no la comparació
    posicional directa.
    """
    retained  = [p for p in preds if p["confidence"] >= threshold]
    discarded = [p for p in preds if p["confidence"] <  threshold]
    return retained, discarded


# ══════════════════════════════════════════════════════════════════════════════
# Avaluació d'un grup complet
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_group(
    model:        nn.Module,
    group:        dict,
    threshold:    float,
    idx_to_class: dict[int, str],
    device:       torch.device,
    strict_length: bool,
) -> dict | None:
    """
    Pipeline complet per a un grup: inferència → filtratge → comparació GT.

    Escull el GT que minimitza la distància d'edició amb la seqüència filtrada
    (en el cas habitual d'un sol GT per stem no hi ha ambigüitat).

    Retorna None si s'aplica --strict_length i la longitud no coincideix.
    """
    # 1. Predicció de tots els caràcters segmentats
    all_preds: list[dict] = []
    for char_idx, img_path in group["entries"]:
        pred_idx, confidence = predict_char(model, img_path, device)
        all_preds.append({
            "char_idx":    char_idx,
            "img_path":    img_path,
            "pred":        idx_to_class.get(pred_idx, "?"),
            "confidence":  confidence,
            "filtered_out": False,   # s'actualitza al pas 2
        })

    # 2. Filtratge per confiança
    retained, discarded_list = filter_by_confidence(all_preds, threshold)
    for p in discarded_list:
        p["filtered_out"] = True

    pred_str = "".join(p["pred"] for p in retained)

    # 3. Millor GT: minimitza edit_distance respecte la seqüència filtrada
    best_gt = min(group["plates_gt"], key=lambda g: edit_distance(pred_str, g))
    gt_len  = len(best_gt)
    ed      = edit_distance(pred_str, best_gt)

    # 4. Mode estricte: descarta si longitud retinguda ≠ longitud GT
    if strict_length and len(retained) != gt_len:
        log.info(
            f"Grup '{group['group_key']}': descartat (--strict_length) — "
            f"retinguts={len(retained)}, GT='{best_gt}' (len={gt_len})"
        )
        return None

    # 5. Alineació posicional (disponible ÚNICAMENT quan les longituds coincideixen)
    #    Usada per a la matriu de confusió i les mètriques posicionals.
    positional_pairs: list[dict] = []
    if len(retained) == gt_len:
        for pred_info, gt_char in zip(retained, best_gt):
            positional_pairs.append({
                "gt":         gt_char,
                "pred":       pred_info["pred"],
                "confidence": pred_info["confidence"],
                "correct":    gt_char == pred_info["pred"],
            })

    return {
        "group_key":       group["group_key"],
        "stem":            group["stem"],
        "box":             group["box"],
        "gt":              best_gt,
        "pred_str":        pred_str,
        "n_segmented":     len(all_preds),
        "n_after_filter":  len(retained),
        "n_discarded":     len(discarded_list),
        "gt_length":       gt_len,
        "edit_distance":   ed,
        "plate_match":     ed == 0,
        "lengths_match":   len(retained) == gt_len,
        "positional_pairs": positional_pairs,
        "all_preds":       all_preds,   # inclou els filtrats, per al CSV
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mètriques agregades
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(group_results: list[dict]) -> dict:
    """
    Agrega les mètriques de tots els grups avaluats.

    Mètriques principals (distància d'edició, robustes a desalineació):
      char_acc_ed  : 1 - sum(ed) / sum(gt_len)
      plate_acc_ed : fracció de grups amb edit_distance == 0

    Mètriques posicionals (grups on n_after_filter == gt_length):
      char_acc_pos : accuracy caràcter a caràcter sobre els grups alineats
    """
    if not group_results:
        return {}

    n_groups      = len(group_results)
    total_ed      = sum(r["edit_distance"]  for r in group_results)
    total_gt_len  = sum(r["gt_length"]      for r in group_results)
    n_plate_match = sum(r["plate_match"]    for r in group_results)

    char_acc_ed  = max(0.0, 1.0 - total_ed / total_gt_len) if total_gt_len > 0 else 0.0
    plate_acc_ed = n_plate_match / n_groups

    # Mètriques posicionals
    pos_pairs    = [pair for r in group_results for pair in r["positional_pairs"]]
    n_pos_groups = sum(1 for r in group_results if r["lengths_match"])
    char_acc_pos = (
        sum(p["correct"] for p in pos_pairs) / len(pos_pairs)
        if pos_pairs else None
    )

    # Confiança (sobre parells posicionalment alineats)
    conf_ok  = [p["confidence"] for p in pos_pairs if     p["correct"]]
    conf_err = [p["confidence"] for p in pos_pairs if not p["correct"]]

    # Estadístiques de filtratge
    total_segmented   = sum(r["n_segmented"]    for r in group_results)
    total_retained    = sum(r["n_after_filter"] for r in group_results)
    total_discarded   = sum(r["n_discarded"]    for r in group_results)
    n_filtered_groups = sum(1 for r in group_results if r["n_discarded"] > 0)

    # Distribució de confiança de tots els caràcters
    retained_confs  = [p["confidence"] for r in group_results
                       for p in r["all_preds"] if not p["filtered_out"]]
    discarded_confs = [p["confidence"] for r in group_results
                       for p in r["all_preds"] if     p["filtered_out"]]

    return {
        "char_acc_ed":       char_acc_ed,
        "plate_acc_ed":      plate_acc_ed,
        "total_ed":          total_ed,
        "total_gt_len":      total_gt_len,
        "n_groups":          n_groups,
        "n_plate_match":     n_plate_match,
        "char_acc_pos":      char_acc_pos,
        "n_pos_groups":      n_pos_groups,
        "n_pos_pairs":       len(pos_pairs),
        "mean_conf_ok":      float(np.mean(conf_ok))  if conf_ok  else 0.0,
        "mean_conf_err":     float(np.mean(conf_err)) if conf_err else 0.0,
        "total_segmented":   total_segmented,
        "total_retained":    total_retained,
        "total_discarded":   total_discarded,
        "n_filtered_groups": n_filtered_groups,
        "retained_confs":    retained_confs,
        "discarded_confs":   discarded_confs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Matriu de confusió
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion(
    y_true:   list[str],
    y_pred:   list[str],
    classes:  list[str],
    out_path: Path,
) -> list[tuple[str, str, int]]:
    """
    Genera i guarda la matriu de confusió normalitzada per files.
    S'usa ÚNICAMENT sobre els parells posicionalment alineats (longituds coincidents).
    Retorna els 10 parells de confusió més freqüents (fora diagonal).
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n  = len(classes)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) > 0,
            cm / cm.sum(axis=1, keepdims=True),
            0.0,
        )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fracció per fila (classe real)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=6, rotation=90)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Caràcter predit")
    ax.set_ylabel("Caràcter real")
    ax.set_title("Matriu de confusió — caràcters reals (grups alineats, normalitzada per fila)")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    log.info(f"Matriu de confusió guardada a: {out_path}")

    off_diag = [
        (classes[i], classes[j], int(cm[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j and cm[i, j] > 0
    ]
    off_diag.sort(key=lambda x: -x[2])
    return off_diag[:10]


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalua CharCNN sobre caràcters reals segmentats."
    )
    parser.add_argument("--segmented_dir", default="data/chars",
                        help="Directori amb els PNGs segmentats")
    parser.add_argument("--gt_dir",        default="data/raw",
                        help="Directori amb els fitxers GT .txt")
    parser.add_argument("--model_path",    default="models/char_cnn_best.pth",
                        help="Checkpoint del model")
    parser.add_argument("--out_dir",       default="models",
                        help="Directori on es guarden el CSV i la figura")
    parser.add_argument("--synthetic_acc", type=float, default=None,
                        help="Val accuracy sintètica (0-1) per calcular el gap; "
                             "si s'omet, s'usa el valor del checkpoint")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Llindar de confiança per filtrar falsos positius (default: 0.5)")
    parser.add_argument("--strict_length", action="store_true",
                        help="Comportament original: descarta grups on la longitud "
                             "filtrada no coincideix amb el GT")
    args = parser.parse_args()

    segmented_dir = Path(args.segmented_dir)
    gt_dir        = Path(args.gt_dir)
    model_path    = Path(args.model_path)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Pas 1: agrupació ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PAS 1 — Agrupació i associació GT")
    print("=" * 62)

    groups, n_no_gt = group_files(segmented_dir, gt_dir)

    # Quants s'haurien descartat amb la regla antiga de longitud exacta
    # (avaluat ABANS del filtratge per confiança, sobre tots els chars segmentats)
    n_would_discard_old = sum(
        1 for g in groups
        if not any(len(p) == len(g["entries"]) for p in g["plates_gt"])
    )

    print(f"\n  Grups amb GT vàlid        : {len(groups)}")
    print(f"  Descartats (sense GT)     : {n_no_gt}")
    print(f"\n  Comparació vs regla antiga (longitud exacta sense filtrar):")
    print(f"    Haurien estat descartats : {n_would_discard_old}")
    print(f"    S'avaluen ara            : {len(groups)}")

    if not groups:
        print(
            "\nERROR: Cap grup amb GT vàlid. Comprova els fitxers GT i el "
            "directori de caràcters segmentats.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ─── Pas 2: càrrega del model ────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PAS 2 — Càrrega del model")
    print("=" * 62)

    model, class_to_idx, idx_to_class, device, ckpt_val_acc = load_model(model_path)

    synth_acc = args.synthetic_acc if args.synthetic_acc is not None else ckpt_val_acc
    if synth_acc is not None:
        src = "CLI" if args.synthetic_acc is not None else "checkpoint"
        log.info(f"Val accuracy sintètica ({src}): {synth_acc:.2%}")

    # ─── Pas 3: inferència + filtratge + comparació GT ───────────────────────
    print("\n" + "=" * 62)
    print(f"PAS 3 — Inferència + filtratge (threshold={args.conf_threshold})"
          + (" + strict_length" if args.strict_length else ""))
    print("=" * 62)

    group_results: list[dict] = []
    n_strict_discarded = 0

    for group in groups:
        result = evaluate_group(
            model, group, args.conf_threshold,
            idx_to_class, device, args.strict_length,
        )
        if result is None:
            n_strict_discarded += 1
        else:
            group_results.append(result)
            log.info(
                f"Grup '{result['group_key']}': "
                f"{result['n_segmented']} seg → {result['n_after_filter']} retinguts "
                f"(GT='{result['gt']}', ed={result['edit_distance']}, "
                f"{'✓' if result['plate_match'] else '✗'})"
            )

    if not group_results:
        print(
            "\nERROR: Tots els grups han estat descartats (mode --strict_length).\n"
            "Prova sense --strict_length o ajusta --conf_threshold.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ─── Pas 4: mètriques ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PAS 4 — Mètriques")
    print("=" * 62)

    metrics = compute_metrics(group_results)

    # ─── Pas 5: matriu de confusió (parells posicionals) ─────────────────────
    classes_sorted = sorted(class_to_idx, key=lambda c: class_to_idx[c])
    pos_pairs = [pair for r in group_results for pair in r["positional_pairs"]]
    top10: list = []
    if pos_pairs:
        y_true = [p["gt"]   for p in pos_pairs]
        y_pred = [p["pred"] for p in pos_pairs]
        top10 = plot_confusion(y_true, y_pred, classes_sorted,
                               out_dir / "confusion_matrix_real.png")
    else:
        log.warning("Cap grup alineat posicionalment; matriu de confusió no generada.")

    # ─── Pas 6: CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / "eval_real_results.csv"
    fieldnames = [
        "stem", "box", "char_idx", "gt", "pred", "confidence", "correct",
        "filtered_out",
        "n_segmented", "n_after_filter", "gt_length", "edit_distance",
        "n_discarded_lowconf",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in group_results:
            retained_seq = [p for p in r["all_preds"] if not p["filtered_out"]]
            # Pre-mapa GT per als retinguts quan les longituds coincideixen
            retained_gt_map: dict[int, str] = {}
            if r["lengths_match"]:
                for pred_info, gt_char in zip(retained_seq, r["gt"]):
                    retained_gt_map[id(pred_info)] = gt_char

            for pred_info in r["all_preds"]:
                if pred_info["filtered_out"]:
                    gt_char = "?"
                    correct = False
                elif r["lengths_match"]:
                    gt_char = retained_gt_map.get(id(pred_info), "?")
                    correct = gt_char == pred_info["pred"]
                else:
                    gt_char = "?"
                    correct = False

                writer.writerow({
                    "stem":               r["stem"],
                    "box":                r["box"],
                    "char_idx":           pred_info["char_idx"],
                    "gt":                 gt_char,
                    "pred":               pred_info["pred"],
                    "confidence":         f"{pred_info['confidence']:.4f}",
                    "correct":            correct,
                    "filtered_out":       pred_info["filtered_out"],
                    "n_segmented":        r["n_segmented"],
                    "n_after_filter":     r["n_after_filter"],
                    "gt_length":          r["gt_length"],
                    "edit_distance":      r["edit_distance"],
                    "n_discarded_lowconf": r["n_discarded"],
                })
    log.info(f"CSV guardat a: {csv_path}")

    # ─── Resum final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RESUM FINAL — AVALUACIÓ SOBRE CARÀCTERS REALS")
    print("=" * 62)

    print(f"\n── Agrupació ─────────────────────────────────────────────")
    print(f"  Grups avaluats                     : {len(group_results)}")
    if n_strict_discarded:
        print(f"  Descartats (--strict_length)       : {n_strict_discarded}")
    print(f"  Descartats (sense GT)              : {n_no_gt}")
    print(f"  Recuperats vs regla antiga longitud: {n_would_discard_old} grups "
          f"que la regla antiga hauria descartat")

    print(f"\n── Mètriques principals (distància d'edició) ─────────────")
    print(f"  Char Accuracy (ED)    : {metrics['char_acc_ed']:.2%}"
          f"  (ED total={metrics['total_ed']}, chars GT={metrics['total_gt_len']})")
    print(f"  Plate Accuracy (ED=0) : {metrics['plate_acc_ed']:.2%}"
          f"  ({metrics['n_plate_match']}/{metrics['n_groups']} matrícules perfectes)")

    if metrics.get("char_acc_pos") is not None:
        print(f"\n── Mètriques posicionals ({metrics['n_pos_groups']} grups alineats) ──────")
        print(f"  Char Accuracy (posicional) : {metrics['char_acc_pos']:.2%}"
              f"  ({metrics['n_pos_pairs']} caràcters)")

    print(f"\n── Filtratge per confiança (threshold={args.conf_threshold}) ────")
    print(f"  Caràcters segmentats   : {metrics['total_segmented']}")
    print(f"  Retinguts              : {metrics['total_retained']}")
    print(f"  Descartats (baixa conf): {metrics['total_discarded']}")
    print(f"  Grups amb algun descartat : {metrics['n_filtered_groups']}")
    if metrics["retained_confs"]:
        print(f"  Conf. mediana retinguts   : {np.median(metrics['retained_confs']):.3f}")
    if metrics["discarded_confs"]:
        print(f"  Conf. mediana descartats  : {np.median(metrics['discarded_confs']):.3f}")

    print(f"\n── Confiança mitjana (parells alineats) ──────────────────")
    print(f"  Prediccions correctes   : {metrics['mean_conf_ok']:.4f}")
    print(f"  Prediccions incorrectes : {metrics['mean_conf_err']:.4f}")

    if synth_acc is not None:
        gap = synth_acc - metrics["char_acc_ed"]
        print(f"\n── Gap sintètic → real ───────────────────────────────────")
        print(f"  Val accuracy sintètica    : {synth_acc:.2%}")
        print(f"  Char accuracy real (ED)   : {metrics['char_acc_ed']:.2%}")
        print(f"  Gap                       : {gap:+.2%}")
        if gap > 0.10:
            print(
                f"\n  AVIS: Gap > 10 punts ({gap:.1%}).\n"
                "  Es recomana revisar el pipeline d'augmentation: els artefactes\n"
                "  del domini real no estan prou representats al dataset sintètic."
            )

    print(f"\n── Top 10 parells de confusió (real → predit) ────────────")
    if top10:
        for gt_c, pred_c, cnt in top10:
            print(f"  {gt_c} → {pred_c}  ({cnt} vegades)")
    else:
        print("  (no disponible — cap grup alineat)")

    # Errors individuals per caràcter (grups posicionals)
    error_counter: Counter = Counter()
    for r in group_results:
        for pair in r["positional_pairs"]:
            if not pair["correct"]:
                error_counter[(pair["gt"], pair["pred"])] += 1

    if error_counter:
        print(f"\n── Errors per caràcter (fins a 20 més freqüents) ────────")
        print(f"  {'Real':>4}  {'Predit':>6}  {'Vegades':>7}  {'Conf.':>6}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*6}")
        for (gt_c, pred_c), cnt in error_counter.most_common(20):
            conf_vals = [
                pair["confidence"]
                for r in group_results
                for pair in r["positional_pairs"]
                if pair["gt"] == gt_c and pair["pred"] == pred_c
            ]
            print(f"  {gt_c:>4}  {pred_c:>6}  {cnt:>7}  {float(np.mean(conf_vals)):>6.3f}")

    print(f"\n── Fitxers generats ──────────────────────────────────────")
    print(f"  CSV        : {csv_path}")
    if pos_pairs:
        print(f"  Confusió   : {out_dir / 'confusion_matrix_real.png'}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
