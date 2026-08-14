# Auditoria de la migració notebook → paquet `alpr/`

| Camp | Valor |
|---|---|
| **Data** | 2026-06-09 |
| **Abast** | Lectura completa de `alpr/` (24 mòduls), `main.py`, els dos `config.py`, el notebook font del detector i el `segmentador.py` de referència. |
| **Guia** | `docs/migracio-notebooks-a-py.md` |
| **Regla** | Només anàlisi. **Cap canvi de codi aplicat.** Les correccions són suggeriments. |
| **Verificació** | Empírica (execució d'imports/atributs de config), no per memòria. |

> **Resum d'una frase del que esperava trobar** (segons `migracio-notebooks-a-py.md`): un paquet
> net amb `common/` + `detector/` (Sobel-v morfològic) + `segmenter/` (normalize→deskew per
> centroides→binarize→contorns→validate→char_export, sortida 28×28 `vj`) + `ocr/` (CharCNN
> separat, infer amb `load_model`/`predict`) + `reader/read_plate`, orquestrat per `main.py`, amb
> **un únic `config.py`** com a font de números i contractes BGR→ROIs→chars 28×28→(classe,conf)→str.
>
> **Conclusió en una frase:** el **detector** i l'**OCR** s'han migrat amb fidelitat alta, però el
> paquet **no arrenca** per un problema de resolució de `config`, i el **segmentador** i el
> **reader** han perdut o substituït decisions documentades (deskew per centroides, normalització
> d'escala, criteri de no-correcció del reader), cosa que els fa divergir del comportament validat.

---

## 🔴 Crítiques

### C1 — Doble `config.py`: `import config` resol al fitxer equivocat → el pipeline no arrenca

> **✅ RESOLT (2026-06-09).** S'han reescrit els 14 `import config` de `alpr/` a `from alpr import
> config`, s'han afegit els `__init__.py` que faltaven (`alpr/`, `common/`, `ocr/`, `ocr/dataset/`,
> `reader/`) i s'ha **eliminat** el `config.py` de l'arrel (pla HOG+SVM deprecat). Verificat:
> `python main.py --input data/raw/eu1.jpg` arrenca i llegeix `M5XSX` (conf 100%, model 36 classes).

**Ubicació:** tots els submòduls de `alpr/` (`import config`): p. ex.
[`alpr/detector/preprocess.py:11`](../alpr/detector/preprocess.py), [`alpr/segmenter/binarize.py:14`](../alpr/segmenter/binarize.py),
[`alpr/ocr/infer.py:21`](../alpr/ocr/infer.py), [`alpr/common/io.py:12`](../alpr/common/io.py). En canvi
[`main.py:38`](../main.py) fa `from alpr import config`.

**Què passa (verificat empíricament):** des de l'arrel del projecte, `import config` resol a
`/config.py` (l'antic pla **HOG+SVM**), **no** a `alpr/config.py`. Aquest config arrel **no conté**
la majoria d'atributs que els mòduls necessiten:

```
ADAPT_BLOCK, ADAPT_C, MORPH_CLOSE_KERNEL, NORM_H, H_CHAR_MIN_REL, H_CHAR_MAX_REL,
IOU_OVERLAP, OUTPUT_SIZE, CANVAS_SIZE, MARGIN, FORMAT_SORTIDA, ROW_STD_H_MAX,
ROW_STD_CY_MAX, WIDTH_OCC_MIN, OCR_INPUT_SIZE, CONF_THRESHOLD, MODEL_CNN_PATH,
GAUSS_KSIZE, GAUSS_SIGMA, CLOSE_KERNEL, OPEN_KERNEL, AREA_RATIO_MIN/MAX,
ASPECT_RATIO_MIN/MAX, EXTENT_MIN, MIN_WIDTH, MIN_HEIGHT      → *** TOTS ABSENTS ***
```

**Per què és crític:**
- `main.py:181` crida `load_model(None)` → [`infer.py:47`](../alpr/ocr/infer.py) `config.MODEL_CNN_PATH`
  → **`AttributeError`**. El pipeline mor a la càrrega del model, abans de processar res.
- Encara que això es salvés, `detect_boxes()` cridaria `config.GAUSS_KSIZE` → `AttributeError`, i el
  segmentador `config.ADAPT_BLOCK` → `AttributeError`.
- Pitjor encara, hi ha col·lisions **silencioses**: `AR_MIN`/`AR_MAX` **sí** existeixen al config
  arrel però valen `1.5`/`7.0` (aspect ratio de **placa sencera**), mentre [`contours.py:45`](../alpr/segmenter/contours.py)
  els fa servir com a AR de **caràcter** (espera 0.05–0.95). Resultat: si arribés a executar-se,
  `1.5 < w/h < 7.0` és fals per a tot caràcter → **0 caràcters, totes les ROIs rebutjades**.

**Contradiu:** `migracio-notebooks-a-py.md` §0.4 ("`config.py` nou = única font de números… cap mòdul
amb constants màgiques pròpies") i §5 ("Es crea de zero; l'actual també és deprecat"). El `CLAUDE.md`
ja avisava: *"poden estar llegint fitxers de config diferents → Cal unificar."* Confirmat.

**Correcció suggerida:** unificar a un sol config i un sol estil d'import.
```python
# A cada mòdul de alpr/ — canviar:
import config
# per:
from alpr import config
```
I retirar/eliminar el `config.py` de l'arrel (és el pla HOG+SVM, ja deprecat). `main.py` ja usa la
forma correcta. *Decisió a confirmar amb tu:* quin dels dos és la font oficial — l'evidència
(CLAUDE.md, ADRs) diu que **`alpr/config.py`** ho ha de ser.

---

### C2 — Deskew reimplementat amb `minAreaRect`: és el mètode que l'ADR-003 va **rebutjar**

> **✅ RESOLT (2026-06-09).** `alpr/segmenter/deskew.py` reescrit amb **regressió de centroides**
> (`estimate_skew_angle` + passada 1 tolerant; la passada 2 estricta la fa `segment()`), i
> `DESKEW_MIN_CANDIDATS` corregit a **3** (ADR-003, tanca part d'I1). Verificat: `estimate_skew_angle`
> sobre una recta sintètica de +10° retorna +9.93°, amb clamp correcte a >15° i mínim de candidats.
> En viu, el deskew ja estima angles reals (eu10 +2.8°, eu11 +3.4°) on `minAreaRect` donava ≈0°, i les
> lectures **crues** (abans del reader) queden gairebé perfectes:
> | Imatge | Angle | OCR cru | GT |
> |---|---|---|---|
> | eu1 | +0.0° | `M5XSX` (exacte) | M5XSX |
> | eu10 | +2.8° | `WAS666O` (≈, S↔5 O↔0) | WA56660 |
> | eu11 | +3.4° | `BS47O4O` (≈, O↔0) | BS47040 |
>
> ⚠️ Aquestes lectures crues les **corromp el reader (C4)** (`WAS666O`→`W456GGO`): C4 és ara el coll
> d'ampolla. *No s'ha tocat el llindar absolut `h>10` de `filter_geometric` (I2) ni el CLAHE de
> `binarize_adaptive` (fora d'abast).*

**Ubicació:** [`alpr/segmenter/deskew.py:24-37`](../alpr/segmenter/deskew.py).

**Què passa:** el deskew migrat binaritza amb Otsu i estima l'angle amb
`cv2.minAreaRect()` sobre tots els píxels de primer pla. El docstring (línia 6) ho diu obertament:
*"Usa minAreaRect sobre els píxels blancs de la binarització Otsu."*

**Per què és un problema:** l'**ADR-003** documenta que `minAreaRect` global es va **implementar i
descartar** perquè *"el núvol global de punts forma una forma rectangular alineada amb els eixos →
angle ≈ 0° independentment de la inclinació real."* La decisió adoptada és **regressió lineal sobre
els centroides** dels caràcters (`estima_angle` a [`segmentador.py:230`](../notebooks/fase_02/segmentador.py)).
La migració ha tornat exactament a l'alternativa rebutjada → el deskew torna a ser ineficaç (angle ≈0°
gairebé sempre), i en alguns casos pot **empitjorar** el crop.

**Regressió addicional:** el `segmentador.py` de referència fa **dues passades** (passada 1 tolerant
només per estimar l'angle; passada 2 estricta). El migrat fa **una sola passada** i estima l'angle
sobre el crop sencer sense filtrar candidats.

**Correcció suggerida:** portar `estima_angle()` (polyfit sobre centroides) i `rota()` de
`segmentador.py:230-258`, i recuperar l'esquema de dues passades de `segmenta_caixa()`
(`segmentador.py:415-469`). Actualitzar el docstring (que ara afirma el contrari de l'ADR-003).

---

### C3 — Falta la normalització d'escala (`NORM_H`): la binarització es trenca en plaques petites

> **✅ RESOLT (2026-06-09).** Nou mòdul `alpr/segmenter/normalize.py` (`normalitza_escala`), cridat a
> l'inici de `segment()` i `segmenta_caixa()` (amb `H,W` recalculats sobre la imatge normalitzada), i
> `NORM_H` corregit a **96**. Verificat per before/after sobre plaques petites:
> | Imatge | Alçada placa | Abans | Després | GT |
> |---|---|---|---|---|
> | eu1 | 46 px | `M5XSX` ✓ | `M5XSX` ✓ (sense regressió) | M5XSX |
> | eu10 | 18 px | *sense detecció* | `WAS666` (conf 98%) | WA56660 |
> | eu11 | 28 px | `BS474` (4 chars) | `8547OAO` (7 chars) | BS47040 |
>
> Les confusions residuals (`8↔B`, `O↔0`, `S↔5`) són del deskew (C2) i del reader (C4), no de la
> segmentació. *No s'ha tocat el deskew ni la manca de CLAHE a `binarize_adaptive` (fora d'abast).*

**Ubicació:** [`alpr/segmenter/segmenter.py:44-56`](../alpr/segmenter/segmenter.py) (`segment()` i
`segmenta_caixa()`); paràmetre a [`alpr/config.py:44`](../alpr/config.py) (`NORM_H = 48`).

**Què passa:** el pipeline migrat fa `deskew → binarize_adaptive → contorns …` **sense normalitzar
l'escala del crop** abans de binaritzar. El `segmentador.py` de referència normalitza el gris a
`NORM_H` **abans** de detectar (`normalitza_escala`, `segmentador.py:140-149`, cridat a la línia 426).

**Per què és crític:** la decisió **D9** (`docs/segmentador-claude-code.md §2.1`) i l'**ADR-002**
expliquen que les plaques d'aquest dataset són de **18–46 px** i que `adaptiveThreshold(blockSize=31)`
*"és més gran que el crop sencer → binaritza malament i sobre-segmenta."* La mesura documentada: sense
normalitzar, **2/25 crops acceptats** i plaques fragmentades (eu1_box0 → 42 blobs); amb `NORM_H=96`,
**6/25** i `eu1_box6 → "M5XSX"` perfecte. El migrat torna a l'estat trencat.

**Doble problema amb el valor:** a més de no aplicar-se, `alpr/config.py` fixa `NORM_H = 48`, però
tota la documentació de referència (D9, ADR-002, situació-actual) usa **96**. (En aquest moment
`NORM_H` és, de fet, **codi mort**: cap mòdul el llegeix.)

**Correcció suggerida:** afegir el pas de normalització a l'inici de `segment()`/`segmenta_caixa()`
(convertir a gris, `normalitza_escala` a `NORM_H=96`, i operar sobre el gris normalitzat), i posar
`NORM_H = 96` al config.

---

### C4 — El reader força el format espanyol 4+3 i **corromp** matrícules no-espanyoles

> **✅ RESOLT (2026-06-09).** `alpr/reader/reader.py` reescrit en **català** com a **muntador pur**
> (neteja + concatena), **sense cap correcció contextual**.
>
> ⚠️ **Aclariment de l'usuari que invalida l'ADR-007:** les matrícules del dataset **no segueixen cap
> patró**. Poden ser de qualsevol país europeu (Sèrbia, Eslovènia, Turquia…), amb A–Z + 0–9 en
> majúscules, **5–9 caràcters en ordre totalment aleatori**. Cap posició té un tipus esperat → **no es
> pot aplicar cap correcció posicional** (ni l'espanyola 4+3, ni "lletra↔dígit per posició" de
> l'ADR-007): davant d'una `S` no se sap si ha de ser `5`, i corregir a cegues introduiria errors. Per
> tant el reader retorna la millor predicció de l'OCR, només netejada.
>
> Verificat: eu1 `M5XSX`, eu10 `WAS666O`, eu11 `BS47O4O` (= OCR cru, sense degradar; abans el reader
> donava `W456GGO`/`8547OAO`). **L'ADR-007 i `situacio-actual.md` §6 queden desactualitzats** (descriuen
> una correcció contextual que no aplica a aquest dataset) — veure nota sota.

**Ubicació:** [`alpr/reader/reader.py:67-110`](../alpr/reader/reader.py).

**Què passa:** `read_plate()` aplica `correct_spanish_plate()` a **qualsevol** string de 7 caràcters,
forçant **dígits a les posicions 0–3** i **lletres a les 4–6** (`reader.py:73-74`). Si el resultat no
casa amb `\d{4}[A-Z]{3}`, **el retorna igualment** (línia 100).

**Per què és crític:**
- L'**ADR-007** especifica el contrari: *"Si el patró no es reconeix, concatenació directa **sense
  corregir**, per no introduir errors nous."* El dataset **barreja formats** (la doc ho repeteix:
  *"sense format fix"*); els exemples reals **no** són 4+3: `M5XSX`, `WA56660`, `PP587AO`, `4BO4979`.
- Exemple de corrupció: GT `WA56660` (7 chars) → força pos 0–3 a dígit (`W,A→4,5,6`) i pos 4–6 a
  lletra (`6→G,6→G,0→O`) → **`W456GGO`**, una lectura pitjor que la concatenació directa. El disseny
  actiu **degrada** matrícules que l'OCR havia encertat.
- A més, hi ha **parells de confusió no previstos a l'ADR-007** (`Q,D→0`, `L,T→1`, `G→6`, `A→4`) que
  augmenten el risc de corrupció.

**Problema secundari (idioma):** tot el mòdul té docstrings i comentaris **en castellà**
(`reader.py:1-10, 39, 50, 59, 68…`), violant la regla d'or del projecte (`convencions-codi.md §1`:
comentaris i docstrings **en català**). És l'únic mòdul de `alpr/` en castellà.

**Correcció suggerida:** reimplementar segons l'ADR-007 — correcció **posicional només quan el patró
s'infereix amb confiança**; si no, retornar `clean_text(chars)` **sense** forçar dígits/lletres.
Limitar els parells de confusió als documentats (O↔0, I↔1, S↔5, Z↔2, B↔8). Traduir-ho tot al català.

---

### ✅ Verificació explícita — Train/test skew del 28×28 (NO és un bug)

S'ha comprovat el risc més perillós citat a l'enunciat. La sortida `vj`
([`char_export.py:39-45`](../alpr/segmenter/char_export.py)) fa *tight crop del binari + `resize(28,28,
INTER_AREA)`*, idèntica a `crops_and_resize` del notebook `from_vj` i coherent amb el format `vj`
adoptat a l'**ADR-005** (67.5% vs 37.5% — el guanyador empíric). El `_TRANSFORM` d'inferència
([`infer.py:27-32`](../alpr/ocr/infer.py)) replica el de l'entrenament ([`train.py:41-45`](../alpr/ocr/train.py))
i el `Resize(28,28)` extra és un no-op. **No hi ha divergència silenciosa** en aquest punt.
*(Matís: `FORMAT_SORTIDA` es llegeix de config; sota el bug C1 seria `AttributeError`.)*

---

## 🟠 Importants

### I1 — Valors de `alpr/config.py` divergeixen del `segmentador.py` de referència

> **✅ RESOLT (2026-06-09) — junt amb I2 i M2.** Com que les divergències de `H_CHAR_*_REL` no es
> podien corregir sense restaurar les **dues etapes** del filtre de la referència (que `alpr` havia
> fusionat), s'ha reescrit `contours.filter_geometric(bboxes, h_placa, strict)` perquè sigui el filtre
> groller **relatiu a l'alçada del crop** (treu l'absolut `h>10` → I2; aplica el filtre d'àrea → M2),
> s'ha afegit `select_dominant_row` (banda de mediana), i `segment()`/`segmenta_caixa()` i `deskew()`
> els encadenen. Valors alineats amb la referència: `H_CHAR_MIN_REL=0.30`, `MAX_REL=1.05`, `AR_MAX=1.10`,
> `H_MED_TOL=0.30`, `AREA_MIN_REL=0.015` (NORM_H i DESKEW_MIN_CANDIDATS ja es van fer a C3/C2;
> `ROW_STD_*` es deixen a 0.20/0.25 perquè coincideixen amb l'ADR-006).
>
> ✅ **Impacte net MESURAT (net-positiu).** Amb l'arnès d'avaluació (I3) sobre tot el dataset (88 ROIs,
> 612 caràcters GT), el pipeline complet (C1–C4 + I1–I5) mesura **millor que la referència validada**:
> | Mètrica | Referència (doc) | Pipeline actual |
> |---|---|---|
> | Char accuracy (ED) | 67.5% | **73.69%** |
> | Plate accuracy | 27.9% (22/79) | **31.82% (28/88)** |
> | Char accuracy posicional | 87.5% | **89.63%** |
>
> El +11 de detecció (77→88) eren plats reals (la plate accuracy puja), no FPs. Les confusions top
> (`0→O`, `1→7`) són domain gap d'OCR documentat, no segmentació. I1 queda confirmat net-positiu.

**Ubicació:** [`alpr/config.py:44-65`](../alpr/config.py) vs `segmentador.py:82-117`.

| Paràmetre | `alpr/config.py` | `segmentador.py` (ref) | Nota |
|---|---|---|---|
| `NORM_H` | **48** | **96** | veure C3 |
| `H_CHAR_MIN_REL` / `MAX_REL` | 0.85 / 1.15 | 0.30 / 1.05 (groller) | semàntica canviada (veure I2) |
| `AR_MAX` | **0.95** | **1.10** | el migrat és més estricte |
| `DESKEW_MIN_CANDIDATS` | **5** | **3** | l'ADR-003 diu 3 |
| `H_MED_TOL` | 0.85 | 0.30 | el de `alpr` és **codi mort** (cap mòdul el llegeix) |
| `ROW_STD_H_MAX` / `CY_MAX` | 0.20 / 0.25 | 0.22 / 0.22 | l'ADR-006 diu 0.20/0.25 — OK respecte l'ADR |
| `AREA_MIN_REL` | 0.0 ("no usat") | 0.015 | el migrat **perd** el filtre d'speckle (veure M4) |

Aquests valors determinen quines ROIs s'accepten; cal alinear-los amb la font validada o documentar
per què canvien.

### I2 — `filter_geometric` usa un llindar **absolut** `h > 10 px` i canvia la semàntica del filtre

> **✅ RESOLT (2026-06-09)** junt amb I1: `filter_geometric` és ara relatiu a `h_placa` (sense `h>10`
> absolut) i la fila dominant es selecciona en un pas separat (`select_dominant_row`).

**Ubicació:** [`alpr/segmenter/contours.py:38`](../alpr/segmenter/contours.py).
```python
heights = [h for (_, _, _, h) in bboxes if h > 10]   # ← px absoluts
```
Viola la regla *"tots els llindars geomètrics relatius a l'alçada de la placa, mai px absoluts"*
(`CLAUDE.md`, `convencions-codi.md §7`, `ADR-002`). És herència del notebook `02_character_v2`.

A més, el migrat filtra l'alçada **relativa a la mediana dels blobs** (`h_med * H_CHAR_MIN_REL`),
mentre que la referència aplica primer un filtre groller **relatiu a l'alçada del crop** (`h_placa`)
i *després* una `selecciona_fila_per_mediana()` separada. Els noms `H_CHAR_MIN/MAX_REL` es reusen amb
un **significat diferent** del de la referència.

**Correcció:** derivar el llindar mínim de l'alçada del crop (p. ex. `H_CHAR_MIN_REL * H`), no d'un
`10` fix, i recuperar el pas de selecció de fila per mediana.

### I3 — Passos i mòduls del pla de migració **no implementats**

> **✅ RESOLT (2026-06-09).** Tots els ítems coberts: `normalize.py` (C3), selecció de fila dominant
> i dues passades del deskew (I1/C2), i ara **`alpr/segmenter/projection.py`** (`detecta_projeccio`,
> activable amb `segment(..., metode="projeccio")`) i **`alpr/ocr/eval_real.py`** (avaluador char/plate
> accuracy ED + posicional + matriu de confusió, reutilitzant `infer.py`). Verificat: el mètode
> projecció dona `M5JKSDK` vs `M5XSX` dels contorns (confirma l'ADR-002), i l'avaluador mesura el
> pipeline complet (veure I1).

Segons `migracio-notebooks-a-py.md` §1–§2, faltaven:

- **`segmenter/normalize.py`** (`normalitza_escala`) — absent (causa de C3).
- **`segmenter/projection.py`** (`detecta_projeccio`, mètode alternatiu activable per flag) — absent.
  El pla i el `CLAUDE.md` del mòdul el demanen explícitament per *"valorar solucions"*.
- **Selecció de fila dominant** (`selecciona_fila_per_mediana`) — absent.
- **Esquema de dues passades** del deskew — absent (veure C2).
- **`ocr/eval_real.py`** (`edit_distance`, `compute_metrics`, `plot_confusion`) — absent; sense ell no
  es poden reproduir les mètriques char/plate accuracy de `situacio-actual.md`.

Cap és un bug en si, però són **incompliments del pla**; llisto'ls perquè decideixis si entren a
l'abast o es documenten com a desviacions conscients.

### I4 — Detector fidel però **inabastable** sota el bug de config

> **✅ RESOLT (2026-06-09).** El detector ja és abastable (C1 resolt). I el text de l'ADR-001 que
> esmentava una "restricció per projecció de files" inexistent s'ha corregit amb una nota
> d'implementació. El codi del detector ja era correcte i no s'ha tocat.

**Ubicació:** `alpr/detector/*`. Comparat **verbatim** amb `notebooks/fase_01/01_morphologic_VJ.ipynb`
(cel·les 5/7/9/11/13): `preprocess`, `sobel_vertical_binary`, `morphology_and_label` i
`filter_by_shape` són **idèntics**, i els valors de `alpr/config.py` (GAUSS (5,5)/1.0, CLOSE (15,3),
OPEN (5,5), AREA 0.001–0.10, ASPECT 1.5–9.0, EXTENT 0.25, MIN_W 40, MIN_H 10) **coincideixen** amb el
notebook. La migració del detector és correcta; l'únic problema és que els seus atributs viuen a
`alpr/config.py` i el runtime carrega el config arrel (C1).

*A confirmar:* l'ADR-001 menciona *"restricció de la cerca a la franja vertical (projecció de files)"*;
**ni el notebook ni el `.py`** la implementen → és el **text de l'ADR** el que sobra, no codi que
falti. No és regressió.

### I5 — `is_plausible_plate`: paràmetre `img_h` no usat

> **✅ RESOLT (2026-06-09).** Eliminat el paràmetre `img_h` (la signatura és ara
> `is_plausible_plate(bboxes, img_w)`) i actualitzades les crides a `segment()`/`segmenta_caixa()`.

**Ubicació:** [`alpr/segmenter/validate.py:19`](../alpr/segmenter/validate.py) (`img_h` marcat
"reservat"). Trencament menor de contracte: la signatura suggereix comprovació vertical que no existeix.
La cobertura es calcula només amb `img_w`. Coherent amb la referència, però documenta la intenció o
elimina el paràmetre.

---

## 🟡 Menors (olors de codi / neteja)

### M1 — Comentaris de nombre de classes incorrectes

> **✅ RESOLT (2026-06-09).** `config.py` → `# 36 classes (A–Z + 0–9)`; `model.py` docstring → `(36
> classes: A–Z, 0–9)`.
- [`alpr/config.py:74`](../alpr/config.py): `CHARS = list("…0-9")  # 32 classes` → en realitat són **36**.
- [`alpr/ocr/model.py:26`](../alpr/ocr/model.py): docstring *"(62 classes)"* → **36**.
- Funcionalment inofensiu (`n_classes` es deriva dinàmicament de `class_to_idx`, `train.py:200`), però
  perpetua la confusió 32/36/62 que els ADRs volien tancar. Corregir a **36 (A–Z, 0–9)**.

### M2 — `AREA_MIN_REL` desactivat → es perd el filtre d'speckle

> **✅ RESOLT (2026-06-09)** junt amb I1: `AREA_MIN_REL=0.015` i `filter_geometric` ara aplica el filtre
> d'àrea (`àrea ≥ AREA_MIN_REL·h_placa²`).
[`alpr/config.py:56`](../alpr/config.py) `AREA_MIN_REL = 0.0` i `contours.py` **no aplica cap filtre
d'àrea**. La referència descarta blobs amb `àrea < 0.015·H²` (`segmentador.py:104,295`). Sense això,
soroll petit pot colar-se com a caràcter.

### M3 — Rutes de `alpr/config.py` relatives al cwd

> **✅ RESOLT (2026-06-09).** Rutes ancorades a `ROOT_DIR = Path(__file__).resolve().parents[1]`.
> Verificat: funcionen executant des d'un cwd diferent (`/tmp`).
[`alpr/config.py:11-18`](../alpr/config.py) usa `Path("data/raw")` (relatiu), mentre el config arrel i
els scripts de referència ancoren a `Path(__file__).parent`. Si s'executa des d'un altre directori,
les rutes (model, dades) es trenquen. Ancorar-les a l'arrel del paquet.

### M4 — Fitxers `__init__.py` solts a l'arrel i a `notebooks/`

> **✅ RESOLT (2026-06-09).** Eliminats tots dos (`__init__.py` arrel via `git rm`, `notebooks/__init__.py`
> via `rm`). `alpr/` segueix sent un paquet vàlid.
[`__init__.py`](../__init__.py) (arrel) i [`notebooks/__init__.py`](../notebooks/__init__.py) buits;
converteixen l'arrel/notebooks en paquets sense necessitat i poden interferir amb la resolució
d'imports. Revisar si calen.

### M5 — `main.py` continua important `cv2.imshow`/`waitKey` per a `--debug`/`--show`

> **✅ RESOLT (2026-06-09).** El bloc de visualització es protegeix amb `try/except cv2.error`: en
> entorns sense display registra un avís i continua el lot en lloc de petar.
[`main.py:242-244`](../main.py) obre finestres OpenCV; correcte per a ús interactiu, però recorda que
trencarà en entorns sense display (headless/CI). No és bug, és una limitació a documentar.

---

## Taula resum

| # | Sev. | Mòdul | Una línia |
|---|---|---|---|
| C1 | ✅ | `config` / tot `alpr/` | ~~`import config` resol al config arrel → pipeline no arrenca~~ **RESOLT**: imports a `from alpr import config`, `__init__.py` afegits, `config.py` arrel eliminat |
| C2 | ✅ | `segmenter/deskew` | ~~Usa `minAreaRect`, rebutjat per l'ADR-003~~ **RESOLT**: regressió de centroides + passada tolerant; `DESKEW_MIN_CANDIDATS=3`; estima angles reals (eu10 +2.8°, eu11 +3.4°) |
| C3 | ✅ | `segmenter/segmenter` | ~~Falta normalització d'escala (`NORM_H`); binarització trencada en plaques petites~~ **RESOLT**: nou `normalize.py`, cridat a `segment()`/`segmenta_caixa()`, `NORM_H=96`; eu10 18px passa de *res* a `WAS666` |
| C4 | ✅ | `reader` | ~~Força format ES 4+3 i corromp matrícules no-ES~~ **RESOLT**: reescrit com a muntador pur sense correcció (el dataset no té patró: 5–9 chars aleatoris EU); eu11 deixa de corrompre's. ADR-007 queda desactualitzat |
| I1 | ✅ | `config` | ~~Valors divergents de la referència~~ **RESOLT** (amb I2+M2): filtre de 2 etapes restaurat, valors alineats. Mesurat net-positiu (char ED 73.7%, plate 31.8% > referència) |
| I2 | ✅ | `segmenter/contours` | ~~`h>10 px` absolut + semàntica fusionada~~ **RESOLT**: `filter_geometric` relatiu a `h_placa` + `select_dominant_row` separat |
| I3 | ✅ | `segmenter` / `ocr` | ~~Falten mòduls del pla~~ **RESOLT**: afegits `projection.py` (+ flag `metode`) i `ocr/eval_real.py`; la resta ja amb C2/C3/I1 |
| I4 | ✅ | `detector` | ~~Inabastable sota C1~~ **RESOLT**: abastable (C1) + text d'ADR-001 corregit; codi del detector ja correcte |
| I5 | ✅ | `segmenter/validate` | ~~`img_h` no usat~~ **RESOLT**: eliminat el paràmetre; `is_plausible_plate(bboxes, img_w)` |
| M1 | ✅ | `config` / `ocr/model` | ~~Comentaris "32"/"62" classes~~ **RESOLT**: corregits a 36 |
| M2 | ✅ | `segmenter/contours` | ~~`AREA_MIN_REL=0` → es perd el filtre d'speckle~~ **RESOLT** (amb I1): `0.015`, aplicat a `filter_geometric` |
| M3 | ✅ | `config` | ~~Rutes relatives al cwd~~ **RESOLT**: ancorades a `ROOT_DIR` |
| M4 | ✅ | arrel / `notebooks` | ~~`__init__.py` buits sospitosos~~ **RESOLT**: eliminats |
| M5 | ✅ | `main` | ~~`cv2.imshow` trenca en headless~~ **RESOLT**: protegit amb `try/except cv2.error` |

---

## Recomanació d'ordre

1. **C1** (config) — desbloqueja tota execució; res es pot provar fins que es resolgui.
2. **C3** + **C2** (normalització + deskew) — recuperen el comportament validat del segmentador.
3. **C4** (reader) — evita corrompre lectures correctes.
4. **I1/I2** (alinear valors i llindars relatius), després **M1–M4** (neteja).
5. **I3** (mòduls del pla) segons decideixis l'abast.

> Cap correcció s'ha aplicat. Digues-me si vols que ataqui alguna d'aquestes troballes (suggeriria
> començar per C1, que és barata i ho desbloqueja tot) i si vull que les converteixi en canvis reals.
