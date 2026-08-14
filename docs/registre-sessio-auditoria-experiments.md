# Registre de sessió — Auditoria de migració, correccions i experiments

| Camp | Valor |
|---|---|
| **Data** | 2026-06-09 |
| **Branca** | `implementation` |
| **Abast** | Auditar la migració `notebooks/` → `alpr/`, corregir les troballes, muntar la infraestructura d'experiments i d'anàlisi de resultats. |
| **Documents germans** | [`auditoria-migracio.md`](auditoria-migracio.md) (informe detallat de troballes), [`situacio-actual.md`](situacio-actual.md), ADRs |

> Aquest fitxer és el **registre narratiu** de tot el que es va fer a la sessió: el perquè de cada
> decisió, els resultats mesurats i l'inventari complet de canvis. Per al detall de cada troballa amb
> ubicació de codi, vegeu `auditoria-migracio.md`.

---

## 1. Resum executiu

Es va partir d'un paquet `alpr/` migrat des dels notebooks que **no arrencava**. La sessió:

1. **Va auditar** `alpr/` + `main.py` contra els notebooks de referència → 14 troballes (4 crítiques, 5 importants, 5 menors).
2. **Va resoldre-les totes**, verificant cada correcció amb mesures reals.
3. **Va corregir un malentès conceptual** sobre el reader (les matrícules no segueixen cap patró).
4. **Va construir l'arnès d'avaluació** (`alpr/ocr/eval_real.py`) que faltava, i amb ell va **mesurar** el pipeline complet.
5. **Va muntar `scripts/experiments/`** amb comparacions de detectors, ablacions del segmentador i atribució d'errors.
6. **Va afegir l'export tidy + plots** per a l'anàlisi estadística posterior.

**Resultat global:** el pipeline migrat passa de no funcionar a mesurar **millor que la referència documentada** (char accuracy ED 67.5% → **73.7%**, plate accuracy 27.9% → **31.8%**).

---

## 2. Auditoria de migració — troballes i estat

Totes resoltes. Detall amb ubicacions a [`auditoria-migracio.md`](auditoria-migracio.md).

| # | Sev. | Mòdul | Problema → resolució |
|---|---|---|---|
| **C1** | 🔴→✅ | `config` / tot `alpr/` | `import config` resolia al `config.py` de l'arrel (HOG+SVM antic) sense els atributs → pipeline no arrencava. **Fix:** tots els submòduls a `from alpr import config`, `__init__.py` afegits, `config.py` arrel eliminat. |
| **C2** | 🔴→✅ | `segmenter/deskew` | Usava `minAreaRect` (mètode rebutjat per l'ADR-003, dona angle ≈0°). **Fix:** regressió de centroides + passada tolerant; `DESKEW_MIN_CANDIDATS=3`. |
| **C3** | 🔴→✅ | `segmenter` | Faltava la normalització d'escala (`NORM_H`); les plaques de 18–46 px es trencaven. **Fix:** nou `normalize.py`, cridat a `segment()`; `NORM_H=96`. |
| **C4** | 🔴→✅ | `reader` | Forçava el format espanyol 4+3 i **corrompia** matrícules no-ES (`WAS666O`→`W456GGO`). **Fix:** muntador pur sense correcció (veure §3). |
| **I1** | 🟠→✅ | `config` | Valors divergents de la referència. **Fix:** restaurat el filtre de dues etapes (`filter_geometric` relatiu a `h_placa` + `select_dominant_row`), valors alineats. |
| **I2** | 🟠→✅ | `segmenter/contours` | Llindar absolut `h>10 px`. **Fix:** tot relatiu a `h_placa`. |
| **I3** | 🟠→✅ | `segmenter`/`ocr` | Mòduls del pla absents. **Fix:** `projection.py` (+ flag `metode`) i `ocr/eval_real.py`. |
| **I4** | 🟠→✅ | `detector` | Inabastable sota C1. **Fix:** abastable (C1) + text d'ADR-001 corregit. |
| **I5** | 🟠→✅ | `segmenter/validate` | Paràmetre `img_h` no usat. **Fix:** eliminat. |
| **M1** | 🟡→✅ | `config`/`model` | Comentaris "32"/"62" classes → corregits a **36**. |
| **M2** | 🟡→✅ | `contours` | `AREA_MIN_REL=0` (filtre d'speckle desactivat) → `0.015`, aplicat. |
| **M3** | 🟡→✅ | `config` | Rutes relatives al cwd → ancorades a `ROOT_DIR`. |
| **M4** | 🟡→✅ | arrel/`notebooks` | `__init__.py` solts → eliminats. |
| **M5** | 🟡→✅ | `main` | `cv2.imshow` peta en headless → protegit amb `try/except`. |

**Verificació destacada:** es va comprovar explícitament que **no hi ha train/test skew** al 28×28 — el format `vj` (default) és coherent entre el segmentador i l'entrenament de la CNN.

---

## 3. Correcció conceptual del reader (important)

La implementació original (i l'ADR-007) assumien que es podia corregir confusions O↔0, S↔5… segons
la **posició dins d'un patró**. L'usuari va aclarir que **aquest dataset no té cap patró**: matrícules
de qualsevol país europeu, A–Z + 0–9 en majúscules, **5–9 caràcters en ordre totalment aleatori**.

**Conseqüència:** no es pot inferir el tipus de cap posició, de manera que **qualsevol correcció a
cegues introdueix errors**. El reader es va reescriure com a **muntador pur** (neteja + concatena la
predicció de l'OCR). Es van actualitzar **ADR-007** (revisat, correcció contextual passa a alternativa
descartada), `situacio-actual.md §6`, `index-memoria.md §4.4`, `migracio-notebooks-a-py.md` i
`ADR-004`. Fet desat a memòria de projecte ([`matricules-sense-patro`]).

---

## 4. Mètriques del pipeline (mesura agregada)

Mesurat amb `python -m alpr.ocr.eval_real` sobre el dataset complet (88 ROIs acceptades, 612 caràcters GT):

| Mètrica | Referència (doc) | **Pipeline actual** |
|---|---|---|
| Char accuracy (ED) | 67.5% | **73.69%** |
| Plate accuracy (ED=0) | 27.9% (22/79) | **31.82% (28/88)** |
| Char accuracy posicional | 87.5% | **89.63%** |

Errors top (matriu de confusió): `0→O` (10), `1→7` (9) — **domain gap d'OCR** documentat, no segmentació.

---

## 5. Experiments (`scripts/experiments/`)

Tots mesurats amb la **mateixa metodologia** (regenerar caràcters amb el pipeline real → avaluar amb
`eval_real`), evitant el biaix d'avaluacions barrejades que advertia `segmenter_audit.md §1.5`.

### 5.1 Comparació de tècniques de detecció (§4.1.3 / §5.2)
`compare_detectors.py` + `detectors.py` (detectors autocontinguts; el Canny és **còpia** de `main_3.py`
perquè aquell fitxer es retirarà).

| Tècnica | Recall (centre) | Recall (IoU≥0.5) | Cand./imatge (mitj/màx) |
|---|---|---|---|
| **morfològic** (oficial) | **98.1%** | **79.6%** | 3.5 / 11 |
| canny multiescala | 73.1% | 61.1% | 4.8 / 5 |
| harris + density | 82.4% | 38.9% | 6.0 / 27 |

→ El morfològic guanya en recall **i** en cost. Harris detecta on hi ha cantonades però localitza
malament (IoU 38.9%) i genera molts FP. **Justifica l'ADR-001.** (HOG+SVM queda bloquejat: sense pesos.)

### 5.2 Ablacions del segmentador (§4.2.3 / §4.2.6 / §5.4)
`ablate_segmenter.py`:

- **E1 — Mètode:** contorns **73.69%** char ED vs projecció **3.51%** (33 ROIs) → confirma l'ADR-002.
- **E2 — Format de sortida** (amb `regla-or` **fidel**, Grup D sobre grisos): `vj` **73.69%** vs
  `regla-or` **38.56%** → **confirma l'ADR-005** (38.56% ≈ 37.7% documentat). *Nota: una primera mesura
  amb un `regla-or` simplificat donava "empat" enganyós; en implementar-lo fidel reapareix el gap.*
- **E6 — Llindar de confiança:** òptim a **conf=0.5** (per sobre descarta caràcters reals).

### 5.3 Ablacions internes (`ablate_pipeline.py`)
- **E3 — `NORM_H`:** punt dolç a **96** (88 ROIs, plate 31.8%); `NORM_H=32` catastròfic (char 58.8%).
  Valida C3/D9.
- **E4 — Deskew:** **amb prou feines importa** (dataset gairebé horitzontal); centroides marginalment
  el millor (més ROIs + millor plate/posicional). Consistent amb l'auditoria (§2.4).

### 5.4 Sensibilitat del detector (`sweep_detector.py`)
- **E5:** `ASPECT_RATIO_MAX` és el knob **crític** (5.0 → recall 70.4%); `EXTENT_MIN` és suau. Justifica
  els defaults (9.0 / 0.25).

### 5.5 Atribució d'errors end-to-end (`error_attribution.py`, §6)
- **E9 — funnel:** detectades **98%** → segmentades **73%** → perfectes **26%**.

| Categoria | Imatges | % |
|---|---|---|
| ✅ perfecta | 28 | 25.9% |
| ❌ error d'OCR | 51 | 47.2% |
| ❌ rebuig del segmentador | 27 | 25.0% |
| ❌ el detector falla | 2 | 1.9% |

→ El coll d'ampolla és clarament l'**OCR (47%, domain gap)**, després el **rebuig del segmentador (25%)**.

---

## 6. Eines d'anàlisi de resultats

- **`export_analysis.py`** — regenera els resultats i exporta dos CSV tidy a `output/experiments/`:
  - `analysis_plates.csv` (1 fila/matrícula): gt, pred, gt_length, n_segmented, n_after_filter,
    delta_len, edit_distance, char_acc_ed, plate_match, mean/median/min/max_conf…
  - `analysis_chars.csv` (1 fila/caràcter): pred, confidence, filtered_out, gt, correct, + camps de grup.
- **`plot_analysis.py`** — genera **10 figures** a `output/experiments/plots/` (boxplots de confiança per
  encert, matriu de confusió, top confusions, accuracy per classe, char_acc per longitud, distribució
  d'ED, sobre/sub-segmentació, confiança per matrícula perfecta, scatter confiança vs accuracy).

**Insight clau revelat:** la CNN és **sobre-confiada** — els caràcters incorrectes tenen confiança
mediana ~0.945, gairebé tan alta com els correctes. Per això el filtre de confiança no neteja bé els
falsos positius (concorda amb `segmenter_audit.md §2.1`).

---

## 7. Inventari complet de canvis

### `alpr/` — codi del paquet
**Nous:**
- `alpr/segmenter/normalize.py` (C3) · `alpr/segmenter/projection.py` (I3) · `alpr/ocr/eval_real.py` (I3)
- `__init__.py` a `alpr/`, `alpr/common/`, `alpr/ocr/`, `alpr/ocr/dataset/`, `alpr/reader/` (C1)

**Reescrits:**
- `alpr/segmenter/deskew.py` (C2: regressió de centroides) · `alpr/reader/reader.py` (C4: muntador pur)
- `alpr/segmenter/contours.py` (I1/I2: filtre de dues etapes) · `alpr/segmenter/char_export.py`
  (regla-or fidel + guard)

**Modificats:**
- `alpr/config.py` (C1/C3/I1/M1/M2/M3: valors i rutes) · `alpr/segmenter/segmenter.py` (pipeline,
  flag `metode`, gray per regla-or) · `alpr/segmenter/validate.py` (I5) · `main.py` (M5)
- 14 mòduls amb `import config` → `from alpr import config` (C1)

**Eliminats:**
- `config.py` (arrel, HOG+SVM deprecat) · `__init__.py` (arrel) · `notebooks/__init__.py`

### `scripts/experiments/` — nou directori
`detectors.py`, `compare_detectors.py`, `ablate_segmenter.py`, `ablate_pipeline.py`,
`sweep_detector.py`, `error_attribution.py`, `export_analysis.py`, `plot_analysis.py`

### `docs/`
**Nous:** `auditoria-migracio.md`, `registre-sessio-auditoria-experiments.md` (aquest)
**Modificats:** `adr/ADR-001` (text fals), `adr/ADR-004` (referència), `adr/ADR-007` (reescrit),
`adr/README.md`, `situacio-actual.md`, `index-memoria.md`, `migracio-notebooks-a-py.md`

### Altres
- `.claude/settings.json` (nou): permet `Edit`/`Write` sense prompt.
- Memòria de projecte: `matricules-sense-patro.md`.
- Artefactes regenerats: `models/eval_real_results.csv`, `models/confusion_matrix_real.png`,
  `data/chars/` (poblat), `output/experiments/*.csv` + `plots/*.png`.

---

## 8. Com reproduir-ho

```bash
# Pipeline end-to-end
python main.py --input data/raw/eu1.jpg

# Mètriques sobre caràcters reals
python -m alpr.ocr.eval_real

# Experiments
python scripts/experiments/compare_detectors.py        # tècniques de detecció
python scripts/experiments/ablate_segmenter.py         # E1, E2, E6
python scripts/experiments/ablate_pipeline.py          # E3 (NORM_H), E4 (deskew)
python scripts/experiments/sweep_detector.py           # E5
python scripts/experiments/error_attribution.py        # E9 (funnel)

# Anàlisi
python scripts/experiments/export_analysis.py          # CSV tidy
python scripts/experiments/plot_analysis.py            # 10 figures
```

---

## 9. Què queda obert

- **ADR-005** — es podria afegir la reproducció `vj` 73.69% vs `regla-or` 38.56% com a confirmació
  independent (no fet).
- **Experiments addicionals** possibles: HOG+SVM (cal entrenar-lo), `plate_detector`/`corner_detection`
  v1 com a 4a/5a tècnica, i "augmentation estil vj" (re-generar el dataset sintètic perquè simuli millor
  la sortida `vj` real — ataca el domain gap, que E9 identifica com el coll d'ampolla).
- **Memòria** — el §5 i §6 ja es poden omplir amb les taules i figures d'aquesta sessió.
