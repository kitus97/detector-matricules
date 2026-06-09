"""
segmentador.py — Segmentador de caràcters de matrícula (Fase 02)
================================================================

Script de proves per **valorar solucions de segmentació**. Recorre els retalls
de possibles matrícules (`{stem}_box{n}.png`) que produeix el detector, en
segmenta els caràcters i en deixa cada caràcter com a imatge 28×28 llesta per a
l'OCR (CNN entrenada a part).

Tot el processament és **visió clàssica** (OpenCV / NumPy): cap model preentrenat,
cap deep learning en la segmentació. La CNN només es consumeix opcionalment al
mode `--ocr` per llegir el resultat.

FORMAT DE LA SORTIDA 28×28 (dos formats seleccionables amb `--format-sortida`)
-----------------------------------------------------------------------------
· "vj" (DEFAULT) — com el notebook `02_character_segmentation_from_vj.ipynb`:
  retall TIGHT del binari de detecció + `cv2.resize((28,28), INTER_AREA)` directe.
  **És el que dona millors resultats d'OCR a la pràctica** (char accuracy ED
  67% vs 38% del format regla-or; veure docs/segmentador-claude-code.md §5).

· "regla-or" — replica la "REGLA D'OR" del CLAUDE.md del mòdul:
  glif en GRISOS centrat en un llenç 64×64 amb marge (`01_render_fonts.py`,
  MARGIN=6, AR preservat) → binarització del "Grup D" de `02_augment.py`
  (`adaptiveThreshold` MEAN_C / INV / 31 / 15) → `resize(28×28, INTER_NEAREST)`.

NOTA IMPORTANT: la teoria de la REGLA D'OR (replicar render+augment) predeia que
"regla-or" hauria de ser superior, però la **mesura empírica amb la CNN real diu el
contrari**: el format "vj" guanya de molt. Com que aquest script serveix per
*valorar solucions*, manem amb les dades, no amb la teoria. Veure el doc §5.

Ús
--
    # Mètode primari (contorns), neteja chars/ i debug/, genera visuals:
    python notebooks/fase_02/segmentador.py

    # Mètode alternatiu (projecció vertical) per comparar:
    python notebooks/fase_02/segmentador.py --metode projeccio

    # Llegir el resultat amb la CNN d'OCR:
    python notebooks/fase_02/segmentador.py --ocr

    # Iteració ràpida sobre uns quants crops, sense esborrar res:
    python notebooks/fase_02/segmentador.py --limit 20 --no-net
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Rutes per defecte (relatives a l'arrel del projecte, sigui quin sigui el cwd)
# ──────────────────────────────────────────────────────────────────────────────
# Aquest fitxer viu a   <arrel>/notebooks/fase_02/segmentador.py
ARREL = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ARREL / "notebooks" / "data" / "processed"   # entrada: crops del detector
CHARS_DIR     = ARREL / "notebooks" / "data" / "chars"       # sortida: caràcters 28×28
DEBUG_DIR     = ARREL / "notebooks" / "data" / "debug"       # sortida: visuals de depuració
MODEL_PATH    = ARREL / "models" / "char_cnn_best.pth"       # checkpoint CNN (mode --ocr)
GT_DIR        = ARREL / "data" / "raw"                        # ground truth (mode --ocr)
TRAIN_SCRIPT  = ARREL / "notebooks" / "ocr" / "03_train_cnn.py"
EVAL_SCRIPT   = ARREL / "notebooks" / "ocr" / "04_evaluate_real.py"

# ══════════════════════════════════════════════════════════════════════════════
# PARÀMETRES CONFIGURABLES (editables aquí dalt per iterar ràpid)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Normalització d'escala del crop ────────────────────────────────────────
# Els crops del detector tenen alçades molt diverses (11–194 px); les plaques
# reals d'aquest dataset són petites (~18–46 px). El `blockSize` de la
# binarització adaptativa està calibrat per a una escala concreta, així que
# normalitzem tots els crops a una alçada de treball comuna abans de detectar.
# Com a bonus, upscalar caràcters minúsculs millora la qualitat del 28×28 final.
NORM_H = 96

# ─── Millora de contrast (CLAHE), només per a la DETECCIÓ ────────────────────
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# ─── Binarització adaptativa per a la DETECCIÓ de contorns ───────────────────
# Mateixos paràmetres que l'ADR-002 i que el Grup D de l'augmentation.
ADAPT_BLOCK = 31          # ha de ser senar
ADAPT_C     = 15
MORPH_CLOSE_KERNEL = (1, 3)   # consolida traços verticals fins fragmentats (la 'I')

# ─── Deskew (correcció d'inclinació) per regressió sobre centroides ──────────
DESKEW_MIN_CANDIDATS = 3      # menys de 3 punts → senyal insuficient, no rota
DESKEW_ANGLE_MAX     = 15.0   # |angle| més gran → es considera detecció errònia

# ─── Filtres geomètrics GROLLERS, RELATIUS A L'ALÇADA DE LA PLACA (H) ─────────
# Mai valors absoluts en px: tot es normalitza per H = alçada del crop.
H_CHAR_MIN_REL = 0.30   # alçada mínima del caràcter / H
H_CHAR_MAX_REL = 1.05   # alçada màxima del caràcter / H
AR_MIN         = 0.05   # aspect ratio (w/h) mínim — NO pujar: mata la 'I' i la '1'
AR_MAX         = 1.10   # aspect ratio màxim — per sobre sol ser soroll o caràcters fusionats
AREA_MIN_REL   = 0.015  # àrea mínima del blob / (H·H)

# ─── Coherència de fila (relatius a la MEDIANA d'alçades dels candidats) ─────
H_MED_TOL      = 0.30   # es conserva el blob si la seva alçada ∈ [0.70, 1.30]·mediana
ROW_STD_H_MAX  = 0.22   # std d'alçades / mediana  (caràcters reals: alçada uniforme)
ROW_STD_CY_MAX = 0.22   # std de cy / mediana d'alçada  (mateixa línia base)
WIDTH_OCC_MIN  = 0.30   # amplada total ocupada / amplada del crop

# ─── Criteri de rebuig (recompte de caràcters vàlids) ────────────────────────
N_CHARS_MIN = 5
N_CHARS_MAX = 9         # NOTA: l'usuari demana [5, 9]; l'ADR-002 deia [5, 8] (veure doc)

# ─── Deduplicació de bboxes solapats (xarxa de seguretat) ────────────────────
IOU_OVERLAP = 0.5

# ─── Sortida 28×28 — REGLA D'OR (replica 01_render_fonts.py + Grup D) ─────────
CANVAS_SIZE = 64        # llenç intermedi (igual que el render de fonts)
MARGIN      = 6         # → el caràcter ocupa CANVAS_SIZE - 2·MARGIN = 52 px (dim. gran)
OUTPUT_SIZE = 28        # mida final que espera la CNN

# ─── Projecció vertical (mètode alternatiu) ──────────────────────────────────
PROJ_SMOOTH_WIN  = 9     # finestra de la mitjana mòbil del histograma de columnes
PROJ_INK_FRAC    = 0.12  # fracció del màxim per considerar una columna "amb tinta"

# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("segmentador")

_FNAME_RE = re.compile(r"^(.+)_box(\d+)\.png$")


# ══════════════════════════════════════════════════════════════════════════════
# Binarització i detecció de contorns
# ══════════════════════════════════════════════════════════════════════════════

def normalitza_escala(gray: np.ndarray) -> np.ndarray:
    """Redimensiona el gris a una alçada de treball comuna (NORM_H), preservant
    l'aspect ratio, perquè el blockSize de la binarització adaptativa tingui
    sempre una escala coherent respecte als caràcters."""
    h, w = gray.shape[:2]
    if h == NORM_H:
        return gray
    nou_w = max(1, int(round(w * NORM_H / float(h))))
    interp = cv2.INTER_CUBIC if h < NORM_H else cv2.INTER_AREA
    return cv2.resize(gray, (nou_w, NORM_H), interpolation=interp)


def binaritza_deteccio(gray: np.ndarray) -> np.ndarray:
    """
    Binaritza el gris per DETECTAR contorns (no per a la sortida final).

    Aplica CLAHE per compensar la il·luminació no uniforme, després
    `adaptiveThreshold` (MEAN_C, BINARY_INV) — robust a gradients d'il·luminació,
    a diferència d'Otsu global — i un MORPH_CLOSE vertical per consolidar la 'I'.
    Retorna binari amb el caràcter BLANC (255) sobre fons NEGRE (0).
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray_eq = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPT_BLOCK,
        C=ADAPT_C,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def detecta_contorns(thresh: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Mètode PRIMARI. `findContours` amb RETR_EXTERNAL (evita el doble contorn
    exterior/forat de 'O','0','D','B'…) i retorna els bounding boxes (x, y, w, h).
    """
    contorns, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contorns]


def detecta_projeccio(thresh: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Mètode ALTERNATIU (flag `--metode projeccio`). Histograma de projecció
    vertical: les bandes de columnes "amb tinta" separades per valls són els
    caràcters. No requereix conèixer el nombre de caràcters a priori.
    """
    col = (thresh > 0).sum(axis=0).astype(np.float64)
    if col.max() <= 0:
        return []
    # Suavitzat amb mitjana mòbil per eliminar el soroll del histograma
    kernel = np.ones(PROJ_SMOOTH_WIN) / PROJ_SMOOTH_WIN
    col_s = np.convolve(col, kernel, mode="same")
    llindar = max(PROJ_INK_FRAC * col_s.max(), 1.0)
    amb_tinta = col_s > llindar

    bboxes: list[tuple[int, int, int, int]] = []
    inici = None
    for i, val in enumerate(amb_tinta):
        if val and inici is None:
            inici = i
        elif not val and inici is not None:
            bb = _bbox_de_banda(thresh, inici, i)
            if bb is not None:
                bboxes.append(bb)
            inici = None
    if inici is not None:
        bb = _bbox_de_banda(thresh, inici, thresh.shape[1])
        if bb is not None:
            bboxes.append(bb)
    return bboxes


def _bbox_de_banda(thresh: np.ndarray, x0: int, x1: int):
    """Donada una banda de columnes [x0, x1), retorna el bbox vertical del seu contingut."""
    banda = thresh[:, x0:x1]
    files = np.where(banda.sum(axis=1) > 0)[0]
    if len(files) == 0:
        return None
    y0, y1 = int(files.min()), int(files.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


# ══════════════════════════════════════════════════════════════════════════════
# Deskew (dues passades) — regressió lineal sobre els centroides dels candidats
# ══════════════════════════════════════════════════════════════════════════════

def estima_angle(bboxes: list[tuple[int, int, int, int]]) -> float:
    """
    Ajusta una recta als centroides (cx, cy) dels candidats i retorna l'angle
    d'inclinació en graus. Els caràcters d'una matrícula estan alineats
    horitzontalment, així que el pendent de la recta revela la inclinació.

    Retorna 0.0 si hi ha menys de DESKEW_MIN_CANDIDATS punts o si |angle| supera
    DESKEW_ANGLE_MAX (senyal insuficient o detecció errònia → no rotar).
    """
    if len(bboxes) < DESKEW_MIN_CANDIDATS:
        return 0.0
    cx = np.array([x + w / 2.0 for x, y, w, h in bboxes])
    cy = np.array([y + h / 2.0 for x, y, w, h in bboxes])
    pendent, _ = np.polyfit(cx, cy, 1)
    angle = float(np.degrees(np.arctan(pendent)))
    if abs(angle) > DESKEW_ANGLE_MAX:
        return 0.0
    return angle


def rota(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rota el gris al voltant del centre. BORDER_REPLICATE evita franges negres als cantons."""
    if abs(angle) < 1e-3:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
# Filtres geomètrics i validació de la caixa
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def elimina_solapats(bboxes, iou_thresh=IOU_OVERLAP):
    """Xarxa de seguretat: si dos bboxes se solapen molt, conserva el de més àrea."""
    ordenats = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
    conservats: list = []
    for bb in ordenats:
        if all(_iou(bb, k) < iou_thresh for k in conservats):
            conservats.append(bb)
    return conservats


def filtra_geometric(bboxes, h_placa, strict=True):
    """
    Filtre GROLLER relatiu a l'alçada de la placa (H). Elimina blobs massa
    petits/grans i amb aspect ratio implausible. En mode tolerant (passada 1,
    per al deskew) eixampla els llindars; en estricte (passada 2) els ajusta.
    """
    h_min = (H_CHAR_MIN_REL * (0.7 if not strict else 1.0)) * h_placa
    h_max = (H_CHAR_MAX_REL * (1.2 if not strict else 1.0)) * h_placa
    ar_max = AR_MAX * (1.3 if not strict else 1.0)
    area_min = AREA_MIN_REL * (h_placa ** 2)

    out = []
    for (x, y, w, h) in bboxes:
        if h <= 0 or w <= 0:
            continue
        if not (h_min <= h <= h_max):
            continue
        ar = w / float(h)
        if not (AR_MIN <= ar <= ar_max):
            continue
        if (w * h) < area_min:
            continue
        out.append((x, y, w, h))
    return out


def coherencia_fila(bboxes, crop_w):
    """
    Comprova que els blobs formen una FILA coherent (caràcters reals d'una
    matrícula). Retorna (és_coherent, motiu_ascii). Llindars relatius a la
    mediana d'alçades, mai px absoluts.
    """
    hs = np.array([h for _, _, _, h in bboxes], dtype=float)
    cys = np.array([y + h / 2.0 for _, y, _, h in bboxes], dtype=float)
    med_h = float(np.median(hs))
    if med_h <= 0:
        return False, "fila_incoherent"
    if hs.std() > ROW_STD_H_MAX * med_h:
        return False, "alcades_inconsistents"
    if cys.std() > ROW_STD_CY_MAX * med_h:
        return False, "baseline_inconsistent"
    xs0 = min(x for x, _, _, _ in bboxes)
    xs1 = max(x + w for x, _, w, _ in bboxes)
    if (xs1 - xs0) / float(crop_w) < WIDTH_OCC_MIN:
        return False, "amplada_insuficient"
    return True, ""


def selecciona_fila_per_mediana(bboxes):
    """De tots els candidats grollers, conserva els que tenen alçada propera a la
    mediana (la fila dominant de caràcters). Robust a alguns blobs de soroll."""
    if not bboxes:
        return []
    med_h = float(np.median([h for _, _, _, h in bboxes]))
    lo, hi = (1 - H_MED_TOL) * med_h, (1 + H_MED_TOL) * med_h
    return [b for b in bboxes if lo <= b[3] <= hi]


# ══════════════════════════════════════════════════════════════════════════════
# Sortida 28×28 — REGLA D'OR (replica render de fonts + Grup D de l'augmentation)
# ══════════════════════════════════════════════════════════════════════════════

def preprocessa_caracter(gray: np.ndarray, bin2: np.ndarray, bbox,
                         format_sortida: str = "regla-or") -> np.ndarray:
    """
    Converteix un caràcter detectat en la imatge 28×28 binària (blanc sobre negre).
    Dos formats seleccionables (flag `--format-sortida`):

    · "regla-or"  → REGLA D'OR del CLAUDE.md (per defecte). Replica
      `01_render_fonts.py` + Grup D de `02_augment.py`: glif en GRISOS centrat en
      un llenç 64×64 amb marge (AR preservat) → adaptiveThreshold(MEAN_C/INV/31/15)
      → resize 28×28 INTER_NEAREST. La textura de vores coincideix amb l'entrenament.

    · "vj"        → com el notebook `02_character_segmentation_from_vj.ipynb`
      (`crops_and_resize`): retall TIGHT del binari de detecció + resize DIRECTE a
      28×28 amb INTER_AREA. Deforma l'aspect ratio i omple vora a vora.
      ATENCIÓ: l'auditoria §2.5 marca aquest patró com a train/test skew.
    """
    x, y, w, h = bbox

    # ── Format "vj": idèntic a crops_and_resize del notebook from_vj ──────────
    if format_sortida == "vj":
        H, W = bin2.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = bin2[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.uint8)
        return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

    # ── Format "regla-or" (per defecte): llenç 64×64 + Grup D ─────────────────
    H, W = gray.shape[:2]
    # 1px de seguretat per no escapçar el contorn, sempre dins de la imatge
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(W, x + w + 1), min(H, y + h + 1)
    glif = gray[y0:y1, x0:x1]
    if glif.size == 0:
        return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.uint8)

    gh, gw = glif.shape[:2]
    objectiu = CANVAS_SIZE - 2 * MARGIN          # 52 px en la dimensió més gran
    escala = objectiu / float(max(gh, gw))
    nou_w = max(1, int(round(gw * escala)))
    nou_h = max(1, int(round(gh * escala)))
    interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
    glif_esc = cv2.resize(glif, (nou_w, nou_h), interpolation=interp)

    # Llenç blanc i enganxat centrat (preserva l'AR; la dim. petita queda amb marge)
    llenc = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, np.uint8)
    off_y = (CANVAS_SIZE - nou_h) // 2
    off_x = (CANVAS_SIZE - nou_w) // 2
    llenc[off_y:off_y + nou_h, off_x:off_x + nou_w] = glif_esc

    # Grup D EXACTE → caràcter BLANC sobre fons NEGRE
    binar = cv2.adaptiveThreshold(
        llenc, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPT_BLOCK,
        C=ADAPT_C,
    )
    # Reducció final a 28×28 (últim pas, INTER_NEAREST com l'augmentation)
    return cv2.resize(binar, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)


# ══════════════════════════════════════════════════════════════════════════════
# Processament d'una caixa
# ══════════════════════════════════════════════════════════════════════════════

def segmenta_caixa(crop_bgr: np.ndarray, metode: str) -> dict:
    """
    Pipeline complet d'una caixa. Retorna un dict amb:
      gray        : gris deskew-at (font per a la sortida 28×28)
      angle       : angle de deskew aplicat (graus)
      detectats   : tots els bboxes detectats a la passada 2 (per visualitzar)
      valids      : bboxes acceptats com a caràcters (ordenats per x)
      acceptada   : bool — si la caixa supera el criteri de rebuig
      motiu       : motiu de rebuig (ascii) o "" si acceptada
    """
    gray0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray0 = normalitza_escala(gray0)
    h0 = gray0.shape[0]

    # ─ Passada 1 (tolerant): estimar l'angle de deskew ─
    bin1 = binaritza_deteccio(gray0)
    cand1 = filtra_geometric(detecta_contorns(bin1), h0, strict=False)
    angle = estima_angle(cand1)

    # ─ Rotació ─
    gray = rota(gray0, angle)
    H = gray.shape[0]

    # ─ Passada 2 (estricta): detecció final amb el mètode triat ─
    bin2 = binaritza_deteccio(gray)
    if metode == "projeccio":
        detectats = detecta_projeccio(bin2)
    else:
        detectats = detecta_contorns(bin2)

    grollers = filtra_geometric(detectats, H, strict=True)
    fila = selecciona_fila_per_mediana(grollers)
    fila = elimina_solapats(fila)
    fila.sort(key=lambda b: b[0])   # esquerra → dreta

    res = {
        "gray": gray, "bin2": bin2, "angle": angle, "detectats": detectats,
        "valids": fila, "acceptada": False, "motiu": "",
    }

    # ─ Criteri de rebuig ─
    n = len(fila)
    if n < N_CHARS_MIN:
        res["motiu"] = f"pocs_caracters(n={n})"
        return res
    if n > N_CHARS_MAX:
        res["motiu"] = f"massa_caracters(n={n})"
        return res
    coherent, motiu = coherencia_fila(fila, gray.shape[1])
    if not coherent:
        res["motiu"] = motiu
        return res

    res["acceptada"] = True
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Visualització de depuració
# ══════════════════════════════════════════════════════════════════════════════

def dibuixa_debug(res: dict, nom: str) -> np.ndarray:
    """
    Dibuixa els bounding boxes sobre la placa deskew-ada. Verd = caràcters
    acceptats; vermell = blobs detectats però descartats. Anota l'angle i, si la
    caixa s'ha rebutjat, el motiu.
    """
    vis = cv2.cvtColor(res["gray"], cv2.COLOR_GRAY2BGR)
    valids = set(map(tuple, res["valids"]))

    # Blobs detectats no acceptats → vermell
    for bb in res["detectats"]:
        if tuple(bb) not in valids:
            x, y, w, h = bb
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # Caràcters acceptats → verd, numerats
    for k, (x, y, w, h) in enumerate(res["valids"]):
        color = (0, 200, 0) if res["acceptada"] else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.putText(vis, str(k), (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Marge superior amb el rètol d'estat
    estat = "ACCEPTADA" if res["acceptada"] else f"REBUTJADA:{res['motiu']}"
    etiqueta = f"{nom} ang={res['angle']:+.1f} n={len(res['valids'])} {estat}"
    vis = cv2.copyMakeBorder(vis, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    col = (0, 255, 0) if res["acceptada"] else (0, 120, 255)
    cv2.putText(vis, etiqueta, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# Mode OCR (opcional) — reutilitza load_model / predict_char de 04_evaluate_real.py
# ══════════════════════════════════════════════════════════════════════════════

def _carrega_modul(path: Path, nom: str):
    """Importa un mòdul des d'un fitxer (els scripts d'OCR comencen per dígit)."""
    spec = importlib.util.spec_from_file_location(nom, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def carrega_ocr():
    """
    Carrega el model CharCNN i les utilitats d'inferència SENSE reinventar res:
    reutilitza `load_model` i `predict_char` de `04_evaluate_real.py` (que al seu
    torn importa la classe CharCNN de `03_train_cnn.py`).

    Retorna (mod_eval, model, idx_to_class, device).
    """
    mod = _carrega_modul(EVAL_SCRIPT, "_eval_real_module")
    model, _class_to_idx, idx_to_class, device, _val_acc = mod.load_model(MODEL_PATH)
    return mod, model, idx_to_class, device


def llegeix_gt(stem: str) -> list[str]:
    """Llegeix les matrícules ground truth de data/raw/{stem}.txt (majúscules)."""
    gt_path = GT_DIR / f"{stem}.txt"
    if not gt_path.exists():
        return []
    plates = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 6:
            plates.append(parts[-1].upper())
    return plates


# ══════════════════════════════════════════════════════════════════════════════
# Utilitats de fitxers
# ══════════════════════════════════════════════════════════════════════════════

def neteja_directori(d: Path) -> int:
    """Esborra els PNG d'un directori (no el directori). Retorna quants n'esborra."""
    if not d.exists():
        return 0
    n = 0
    for f in d.glob("*.png"):
        f.unlink()
        n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Bucle principal
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--chars-dir", type=Path, default=CHARS_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR)
    parser.add_argument("--metode", choices=["contorns", "projeccio"], default="contorns",
                        help="Mètode de segmentació (default: contorns)")
    parser.add_argument("--format-sortida", choices=["regla-or", "vj"], default="vj",
                        help="Format del 28×28: 'vj' (retall binari + resize directe, com el "
                             "notebook from_vj — DEFAULT, millors resultats d'OCR) o 'regla-or' "
                             "(llenç 64×64 + Grup D, REGLA D'OR del CLAUDE.md)")
    parser.add_argument("--ocr", action="store_true",
                        help="Llegeix els caràcters generats amb la CNN d'OCR")
    parser.add_argument("--no-net", action="store_true",
                        help="No esborrar chars/ ni debug/ abans de començar")
    parser.add_argument("--no-debug", action="store_true",
                        help="No generar visuals de depuració")
    parser.add_argument("--limit", type=int, default=0,
                        help="Processa només els N primers crops (0 = tots)")
    args = parser.parse_args()

    args.chars_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    # ─ Neteja (per defecte sí, segons decisió de l'usuari) ─
    if not args.no_net:
        nc = neteja_directori(args.chars_dir)
        nd = neteja_directori(args.debug_dir)
        log.info(f"Neteja: {nc} chars i {nd} visuals esborrats.")

    crops = sorted(args.processed_dir.glob("*_box*.png"))
    if args.limit > 0:
        crops = crops[:args.limit]
    if not crops:
        log.error(f"Cap crop '*_box*.png' a {args.processed_dir}")
        return
    log.info(f"Mètode: {args.metode}  |  Crops a processar: {len(crops)}")

    # ─ Carrega l'OCR si cal ─
    mod_eval = model = idx_to_class = device = None
    if args.ocr:
        log.info("Carregant model d'OCR…")
        mod_eval, model, idx_to_class, device = carrega_ocr()

    n_acceptades = 0
    n_chars_total = 0
    motius = Counter()

    for crop_path in crops:
        m = _FNAME_RE.match(crop_path.name)
        if not m:
            log.warning(f"Nom no reconegut, ignorat: {crop_path.name}")
            continue
        nom = crop_path.stem            # {stem}_box{n}
        stem = m.group(1)

        crop = cv2.imread(str(crop_path))
        if crop is None:
            log.warning(f"No s'ha pogut llegir {crop_path.name}")
            continue

        res = segmenta_caixa(crop, args.metode)

        if not args.no_debug:
            cv2.imwrite(str(args.debug_dir / f"{nom}.png"), dibuixa_debug(res, nom))

        if not res["acceptada"]:
            motius[res["motiu"].split("(")[0]] += 1
            log.info(f"  {nom}: REBUTJADA — {res['motiu']} "
                     f"({len(res['detectats'])} blobs detectats)")
            continue

        # ─ Caixa acceptada: genera i desa els caràcters 28×28 ─
        n_acceptades += 1
        rutes_chars = []
        for k, bb in enumerate(res["valids"]):
            char_img = preprocessa_caracter(res["gray"], res["bin2"], bb, args.format_sortida)
            out_path = args.chars_dir / f"{nom}_char{k}.png"
            cv2.imwrite(str(out_path), char_img)
            rutes_chars.append(out_path)
            n_chars_total += 1

        msg = f"  {nom}: ACCEPTADA — {len(res['valids'])} caràcters (ang={res['angle']:+.1f}°)"

        # ─ Lectura OCR opcional ─
        if args.ocr:
            lectura, confs = [], []
            for p in rutes_chars:
                idx, conf = mod_eval.predict_char(model, p, device)
                lectura.append(idx_to_class.get(idx, "?"))
                confs.append(conf)
            text = "".join(lectura)
            conf_mit = float(np.mean(confs)) if confs else 0.0
            gt = llegeix_gt(stem)
            gt_str = f"  GT={gt[0]}" if gt else ""
            msg += f"  →  OCR='{text}' (conf={conf_mit:.2f}){gt_str}"

        log.info(msg)

    # ─ Resum final ─
    log.info("─" * 60)
    log.info(f"Crops processats     : {len(crops)}")
    log.info(f"Caixes acceptades    : {n_acceptades}")
    log.info(f"Caixes rebutjades    : {len(crops) - n_acceptades}")
    log.info(f"Caràcters generats   : {n_chars_total}  →  {args.chars_dir}")
    if motius:
        log.info("Motius de rebuig:")
        for motiu, cnt in motius.most_common():
            log.info(f"    {motiu:24s}: {cnt}")


if __name__ == "__main__":
    main()
