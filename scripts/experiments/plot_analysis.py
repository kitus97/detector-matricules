"""
scripts/experiments/plot_analysis.py
=====================================
Genera el conjunt complet de figures d'anàlisi dels resultats finals del
pipeline, a partir dels CSV tidy que produeix `export_analysis.py`
(`analysis_plates.csv` i `analysis_chars.csv`).

Figures generades (a output/experiments/plots/):

  Per caràcter:
    01_confianca_per_encert        boxplot  confiança ~ correcte/incorrecte
    02_hist_confianca              histograma de confiança (correctes vs incorrectes)
    03_matriu_confusio             heatmap GT × predicció (grups alineats)
    04_top_confusions              barres dels parells de confusió més freqüents
    05_accuracy_per_classe         barres d'accuracy per caràcter (quins costen més)

  Per matrícula:
    06_characcED_per_longitud      boxplot  char_acc_ed ~ gt_length
    07_hist_edit_distance          histograma de la distància d'edició
    08_delta_len                   sobre/sub-segmentació (n_segmentats − gt_length)
    09_conf_per_plate_match        boxplot  mean_conf ~ matrícula perfecta
    10_scatter_conf_vs_acc         dispersió  mean_conf vs char_acc_ed

Sense dependència de pandas: llegeix els CSV amb la biblioteca estàndard.

Ús
--
    python scripts/experiments/export_analysis.py     # 1r: genera els CSV
    python scripts/experiments/plot_analysis.py       # 2n: genera les figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARREL = Path(__file__).resolve().parents[2]


# ── Càrrega ────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: no existeix {path}. Executa abans export_analysis.py.")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(rows, key):
    """Columna de floats, ignorant buits."""
    return np.array([float(r[key]) for r in rows if r[key] not in ("", None)])


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"{name}.png"), dpi=130)
    plt.close(fig)
    print(f"  → {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figures per caràcter
# ══════════════════════════════════════════════════════════════════════════════

def plots_chars(chars: list[dict], out: Path) -> None:
    aligned = [c for c in chars if c["correct"] in ("0", "1")]
    conf_ok  = [float(c["confidence"]) for c in aligned if c["correct"] == "1"]
    conf_err = [float(c["confidence"]) for c in aligned if c["correct"] == "0"]

    # 01 — boxplot confiança per encert
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([conf_ok, conf_err], tick_labels=["correcte", "incorrecte"], showmeans=True)
    ax.set_ylabel("Confiança (softmax)")
    ax.set_title("Confiança de la CNN segons l'encert\n(grups alineats)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "01_confianca_per_encert")

    # 02 — histograma de confiança
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf_ok,  bins=bins, alpha=0.6, label=f"correctes (n={len(conf_ok)})")
    ax.hist(conf_err, bins=bins, alpha=0.6, label=f"incorrectes (n={len(conf_err)})")
    ax.set_xlabel("Confiança"); ax.set_ylabel("Nombre de caràcters")
    ax.set_title("Distribució de confiança (correctes vs incorrectes)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out, "02_hist_confianca")

    # Parells (gt, pred) dels alineats
    pairs = [(c["gt"], c["pred"]) for c in aligned if c["gt"] != "?"]
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    idx = {c: i for i, c in enumerate(classes)}

    # 03 — matriu de confusió (normalitzada per fila)
    cm = np.zeros((len(classes), len(classes)))
    for g, p in pairs:
        cm[idx[g], idx[p]] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = np.where(cm.sum(1, keepdims=True) > 0, cm / cm.sum(1, keepdims=True), 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fracció per fila (classe real)")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicció"); ax.set_ylabel("Real")
    ax.set_title("Matriu de confusió (grups alineats, normalitzada per fila)")
    _save(fig, out, "03_matriu_confusio")

    # 04 — top parells de confusió
    wrong = Counter((g, p) for g, p in pairs if g != p)
    if wrong:
        top = wrong.most_common(15)
        labels = [f"{g}→{p}" for (g, p), _ in top]
        vals = [n for _, n in top]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(top)), vals, color="#c44")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(labels)
        ax.invert_yaxis(); ax.set_xlabel("Nombre d'errors")
        ax.set_title("Parells de confusió més freqüents (real → predit)")
        ax.grid(axis="x", alpha=0.3)
        _save(fig, out, "04_top_confusions")

    # 05 — accuracy per classe
    per_class = defaultdict(lambda: [0, 0])   # classe → [correctes, total]
    for c in aligned:
        if c["gt"] == "?":
            continue
        per_class[c["gt"]][1] += 1
        per_class[c["gt"]][0] += int(c["correct"] == "1")
    cls = sorted(per_class)
    accs = [per_class[c][0] / per_class[c][1] for c in cls]
    tots = [per_class[c][1] for c in cls]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(cls)), accs, color="#48a")
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per classe de caràcter (núm. = mostres)")
    for b, t in zip(bars, tots):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(t),
                ha="center", va="bottom", fontsize=6)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "05_accuracy_per_classe")


# ══════════════════════════════════════════════════════════════════════════════
# Figures per matrícula
# ══════════════════════════════════════════════════════════════════════════════

def plots_plates(plates: list[dict], out: Path) -> None:
    # 06 — char_acc_ed per longitud de GT
    by_len = defaultdict(list)
    for r in plates:
        by_len[int(r["gt_length"])].append(float(r["char_acc_ed"]))
    lens = sorted(by_len)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([by_len[l] for l in lens], tick_labels=[str(l) for l in lens], showmeans=True)
    ax.set_xlabel("Longitud del GT (caràcters)"); ax.set_ylabel("Char accuracy (ED)")
    ax.set_title("Char accuracy (ED) per longitud de matrícula")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "06_characcED_per_longitud")

    # 07 — histograma d'edit distance
    ed = _f(plates, "edit_distance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ed, bins=range(0, int(ed.max()) + 2), color="#a64", align="left", rwidth=0.85)
    ax.set_xlabel("Distància d'edició"); ax.set_ylabel("Nombre de matrícules")
    ax.set_title("Distribució de la distància d'edició per matrícula")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "07_hist_edit_distance")

    # 08 — delta_len (sobre/sub-segmentació)
    dl = _f(plates, "delta_len").astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    vals, counts = np.unique(dl, return_counts=True)
    colors = ["#c44" if v != 0 else "#4a4" for v in vals]
    ax.bar(vals, counts, color=colors)
    ax.set_xlabel("n_segmentats − gt_length  (0 = exacte)")
    ax.set_ylabel("Nombre de matrícules")
    ax.set_title("Sobre-segmentació (+) vs sub-segmentació (−)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "08_delta_len")

    # 09 — mean_conf per plate_match
    conf_perf = [float(r["mean_conf"]) for r in plates
                 if r["plate_match"] == "1" and r["mean_conf"] != ""]
    conf_imp  = [float(r["mean_conf"]) for r in plates
                 if r["plate_match"] == "0" and r["mean_conf"] != ""]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([conf_perf, conf_imp], tick_labels=["perfecta", "amb error"], showmeans=True)
    ax.set_ylabel("Confiança mitjana de la matrícula")
    ax.set_title("Confiança mitjana segons si la matrícula és perfecta")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "09_conf_per_plate_match")

    # 10 — scatter mean_conf vs char_acc_ed
    mc = np.array([float(r["mean_conf"]) for r in plates if r["mean_conf"] != ""])
    ca = np.array([float(r["char_acc_ed"]) for r in plates if r["mean_conf"] != ""])
    pm = np.array([r["plate_match"] == "1" for r in plates if r["mean_conf"] != ""])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mc[~pm], ca[~pm], alpha=0.6, label="amb error", color="#c44")
    ax.scatter(mc[pm], ca[pm], alpha=0.8, label="perfecta", color="#4a4")
    ax.set_xlabel("Confiança mitjana"); ax.set_ylabel("Char accuracy (ED)")
    ax.set_title("Confiança vs accuracy per matrícula")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out, "10_scatter_conf_vs_acc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, default=ARREL / "output" / "experiments",
                        help="Directori amb analysis_plates.csv i analysis_chars.csv")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="On desar les figures (default: <in-dir>/plots)")
    args = parser.parse_args()

    out = args.out_dir or (args.in_dir / "plots")
    plates = load_csv(args.in_dir / "analysis_plates.csv")
    chars  = load_csv(args.in_dir / "analysis_chars.csv")

    print(f"Generant figures a {out} …")
    plots_chars(chars, out)
    plots_plates(plates, out)
    print(f"\n{10} figures generades.")


if __name__ == "__main__":
    main()
