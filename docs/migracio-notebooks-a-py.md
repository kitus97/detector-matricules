# Guia d'implementació — Migració de notebooks a paquet Python (`.py`)

> Guia per portar el projecte ALPR (detecció + lectura de matrícules) de notebooks a un
> paquet Python clàssic, net i modular. **No conté codi**: és el plànol d'on va cada cosa,
> quines funcions exposa cada mòdul i quins són els contractes entre fases.
>
> ⚠️ **`src/` està DEPRECAT i no s'utilitza res.** Tota la bastida actual de `src/`
> (`pipeline.py`, `reader.py`, `utils.py`, `detector.py`, `ocr.py`, `segmenter.py`,
> `char_segmenter_histogram.py`, `main.py`, `evaluate.py`, el subdirectori `pipeline/`…)
> es considera obsoleta. El paquet nou es construeix **des de zero** a partir de:
>
> - **Fase 1 — Detector:** `notebooks/01_morphologic_VJ.ipynb` (morfològic).
> - **Fase 2 — Segmentador:** `notebooks/fase_02/02_character_segmentation_from_vj.ipynb`
>   + `notebooks/fase_02/segmentador.py` (versió ja madura i validada).
> - **Fase 3 — OCR:** `notebooks/ocr/01_render_fonts.py`, `02_augment.py`,
>   `03_train_cnn.py`, `04_evaluate_real.py` (ja són `.py`).

---

## 0. Principis de la migració

1. **Una funció = una responsabilitat.** Cada cel·la de notebook que fa feina real es
   converteix en funció pura i testejable.
2. **Separar nucli de presentació.** Les cel·les de *visualització* (`matplotlib`) i de
   *bucle de demostració* NO van al nucli del paquet: van a `alpr/viz.py` o als
   `scripts/` d'entrada.
3. **Contractes clars entre fases** (veure §4). Cap fase coneix els detalls interns d'una
   altra; es comuniquen amb tipus ben definits.
4. **`config.py` nou = única font de números i rutes.** Es crea de zero (l'actual també és
   deprecat); cap mòdul amb constants màgiques pròpies.
5. **Convencions del projecte:** comentaris/docstrings/sortides en **català**;
   identificadors en anglès permesos. Imatges **BGR `uint8`** (OpenCV) a tot arreu.
6. **Cap model preentrenat** a detecció/segmentació (només OpenCV/NumPy/SciPy/scikit-image).
   L'OCR és l'única part amb deep learning (CNN entrenada des de zero per l'equip).

---

## 1. Estructura de directoris proposada

Paquet nou `alpr/` (Automatic License Plate Recognition) a l'arrel, independent de `src/`:

```
detector-matricules/
├── config.py                   # NOU. Única font d'hiperparàmetres i rutes
├── pyproject.toml / requirements.txt   # afegir-hi torch/torchvision/Pillow (§7)
│
├── alpr/                       # ── PAQUET PRINCIPAL (nou, des de zero) ──
│   ├── __init__.py
│   │
│   ├── common/                 # utilitats transversals
│   │   ├── __init__.py
│   │   ├── io.py               # load_image(), save_image(), to_grayscale()
│   │   ├── geometry.py         # iou(), nms()
│   │   └── annotations.py      # parse_annotation(), load_annotations(), get_txt_path()
│   │
│   ├── detector/               # FASE 1  ← 01_morphologic_VJ.ipynb
│   │   ├── __init__.py
│   │   ├── preprocess.py       # preprocess()
│   │   ├── edges.py            # sobel_vertical_binary()
│   │   ├── morphology.py       # morphology_and_label()
│   │   ├── shape_filter.py     # filter_by_shape()
│   │   └── detector.py         # detect_boxes(), detect(), detect_debug()   ← API pública
│   │
│   ├── segmenter/              # FASE 2  ← 02_character_segmentation_from_vj.ipynb + segmentador.py
│   │   ├── __init__.py
│   │   ├── normalize.py        # normalitza_escala()
│   │   ├── deskew.py           # estima_angle(), rota()  (deskew per regressió de centroides)
│   │   ├── binarize.py         # binaritza_deteccio()  (adaptiveThreshold + MORPH_CLOSE)
│   │   ├── contours.py         # detecta_contorns(), filtra_geometric(),
│   │   │                       #   selecciona_fila_per_mediana(), elimina_solapats()
│   │   ├── projection.py       # detecta_projeccio()  (mètode alternatiu, opcional)
│   │   ├── validate.py         # coherencia_fila()  (criteri de rebuig 5–9 + coherència)
│   │   ├── char_export.py      # preprocessa_caracter()  → 28×28 format "vj" (§7)
│   │   └── segmenter.py        # segment(), segmenta_caixa(), save_chars()  ← API pública
│   │
│   ├── ocr/                    # FASE 3  ← notebooks/ocr/*.py
│   │   ├── __init__.py
│   │   ├── model.py            # class CharCNN                             ← 03_train_cnn.py
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── render_fonts.py # find_fonts(), fit_font(), render_char(), render_all()  ← 01
│   │   │   └── augment.py      # apply_geometric/stroke/contour, apply_binarize, augment, build_dataset  ← 02
│   │   ├── train.py            # build_dataloaders(), train_one_epoch(), evaluate(), plot_curves()  ← 03
│   │   ├── infer.py            # load_model(), predict(), predict_char()   ← API pública (04)
│   │   └── eval_real.py        # edit_distance(), group_files(), evaluate_group(), compute_metrics(), plot_confusion()  ← 04
│   │
│   ├── reader/                 # FASE 3b  ← concepte de CLAUDE.md (a implementar nou)
│   │   ├── __init__.py
│   │   └── reader.py           # read_plate()  (muntatge directe, sense correcció)
│   │
│   ├── pipeline.py             # orquestrador detect → segment → ocr → reader  ← 04_pipeline.ipynb
│   └── viz.py                  # show_pipeline_stages(), dibuixa_debug(), visualize_plate()
│
├── scripts/                    # punts d'entrada CLI (substitueixen els bucles de notebook)
│   ├── run_detector.py         # demo/avaluació Fase 1
│   ├── run_segmenter.py        # = l'actual notebooks/fase_02/segmentador.py
│   ├── build_ocr_dataset.py    # render_fonts → augment (prep dades OCR)
│   ├── train_ocr.py            # entrena la CNN → models/char_cnn_best.pth
│   ├── evaluate.py             # mètriques (recall detecció, char/plate accuracy ED)
│   └── run_pipeline.py         # end-to-end: data/raw → output/results.csv
│
├── notebooks/                  # ES MANTENEN com a exploració/registre històric
├── models/                     # char_cnn_best.pth, class_to_idx.json, corbes, matriu confusió
├── data/                       # raw/ candidates/ processed/ chars/ ocr_dataset/ synthetic/
├── output/  ·  docs/  ·  resources/fonts/
```

> **Variant minimalista** (si no vols subpaquets): tres fitxers plans
> `alpr/detector.py`, `alpr/segmenter.py`, `alpr/ocr.py` + `alpr/common.py` +
> `alpr/reader.py` + `alpr/pipeline.py`. La taula de funcions de §2 segueix sent vàlida;
> només canvia on viu cada funció.

---

## 2. Inventari de funcions: notebook origen → mòdul destí

### Fase 1 — Detector (`notebooks/01_morphologic_VJ.ipynb`)

Pipeline: `gris → Gaussià → CLAHE → Sobel vertical → Otsu → closing horitzontal → opening
→ components connexos → filtre de forma`. Filosofia de **màxim recall** (millor diversos
falsos positius que perdre la matrícula).

| Funció al notebook | Mòdul destí | Notes de migració |
|---|---|---|
| `preprocess(img_bgr)` | `detector/preprocess.py` | gris → `GaussianBlur` → `CLAHE`. Mou `clipLimit`, `tileGridSize`, `ksize`, `sigma` a `config`. |
| `sobel_vertical_binary(gray)` | `detector/edges.py` | Sobel-x (`CV_16S`) + `convertScaleAbs` + Otsu. Retorna `(sobel_abs, binary)`. |
| `morphology_and_label(binary)` | `detector/morphology.py` | closing `(15,3)` → opening `(5,5)` → `connectedComponentsWithStats`. **Mou les mides de kernel a `config`** (i documenta que depenen de la resolució). |
| `filter_by_shape(num_labels, stats, img_shape)` | `detector/shape_filter.py` | àrea relativa, AR, extent, mida mínima. **Treu els llindars hardcoded a `config`** (`AREA_RATIO_*`, `ASPECT_RATIO_*`, `EXTENT_MIN`, `MIN_WIDTH/HEIGHT`). Retorna `list[(x,y,w,h)]`. |
| `detect_plates(img_bgr) -> dict` | `detector/detector.py` | renombra a **`detect_debug()`** (retorna fases intermèdies per visualitzar). |
| `parse_annotation`, `get_txt_path` | `common/annotations.py` | parseig del GT `filename⇥x⇥y⇥w⇥h⇥text`. |
| `diagnose_failures` | `scripts/evaluate.py` | eina de diagnòstic; va a script, no al nucli. |
| `show_pipeline_stages`, graella, bucle d'execució | `alpr/viz.py` + `scripts/run_detector.py` | NO al nucli. |

**API pública nova del detector** (no existeix tal qual al notebook; cal definir-la):

```python
def detect_boxes(img_bgr) -> list[tuple[int,int,int,int]]   # [(x,y,w,h), ...] en píxels originals
def detect(img_bgr)       -> list[np.ndarray]               # retalls BGR de cada box (per al pipeline)
def detect_debug(img_bgr) -> dict                           # {enhanced, sobel, binary, morph, boxes, n_total}
```

### Fase 2 — Segmentador (`02_character_segmentation_from_vj.ipynb` + `segmentador.py`)

> ⭐ **El `segmentador.py` de `notebooks/fase_02/` ja és la versió madura i validada**
> (inclou normalització d'escala, deskew per regressió, criteri de rebuig [5,9], els dos
> formats de sortida i el mode OCR). La migració és sobretot **reorganitzar-lo** en mòduls;
> el notebook `from_vj` aporta les funcions de referència equivalents.

| Funció (origen) | Mòdul destí | Notes |
|---|---|---|
| `normalitza_escala()` (segmentador.py) | `segmenter/normalize.py` | **imprescindible**: normalitza a `NORM_H` abans de detectar (les plaques del dataset són de 18–46 px i el `blockSize=31` fix falla sense això). |
| `estima_angle()`, `rota()` (segmentador.py) | `segmenter/deskew.py` | deskew per **regressió lineal sobre centroides** (no `minAreaRect` global, que sempre dóna ≈0°). |
| `binaritza_deteccio()` / `binarize_adaptive()` | `segmenter/binarize.py` | CLAHE → `adaptiveThreshold(MEAN_C, INV, 31, 15)` → `MORPH_CLOSE(1,3)` (consolida la 'I'). |
| `detecta_contorns()`, `filtra_geometric()`, `selecciona_fila_per_mediana()`, `elimina_solapats()` | `segmenter/contours.py` | `findContours(RETR_EXTERNAL)` (evita doble contorn de O/0/D); filtres **relatius a l'alçada de placa**; AR mín **0.05** (per la 'I'); dedup IoU. |
| `detecta_projeccio()` | `segmenter/projection.py` | mètode **alternatiu** (projecció vertical) activable per flag, per comparar. |
| `coherencia_fila()` / `is_plausible_plate()` | `segmenter/validate.py` | criteri de rebuig: <5 o >9 caràcters, std d'alçades, alineació de baseline, cobertura horitzontal. Registra el motiu. |
| `preprocessa_caracter()` / `crops_and_resize()` | `segmenter/char_export.py` | sortida **28×28 binari blanc-sobre-negre**, format **"vj"** per defecte (§7). |
| `segmenta_caixa()` | `segmenter/segmenter.py` | orquestra una ROI: normalitza → deskew → binaritza → detecta → filtra → rebuig. |
| `parse_crop_filename()` | `common/annotations.py` o `segmenter` | `{stem}_box{n}` / `{stem}_cand{n}`. |
| `dibuixa_debug()`, `visualize_plate()` | `alpr/viz.py` | visuals de depuració. |
| bucle principal de `segmentador.py` | `scripts/run_segmenter.py` | CLI: `--metode`, `--format-sortida`, `--ocr`, `--limit`, neteja de directoris. |

**API pública nova del segmentador** (la que consumeix el pipeline):

```python
def segment(roi_bgr) -> list[np.ndarray]   # [] si la ROI es rebutja; si no, chars 28×28 binaris
def save_chars(chars, roi_name, out_dir, metadata=None) -> None
```

### Fase 3 — OCR (`notebooks/ocr/*.py`)

> Aquests **ja són scripts `.py`**. La feina és *empaquetar-los* com a mòduls importables i
> **separar la classe del model de l'entrenament** perquè la inferència la pugui importar net
> (avui `04_evaluate_real.py` ha de fer `importlib` perquè `CharCNN` viu dins d'un script que
> comença per dígit — amb el paquet nou això desapareix).

| Funcions (origen) | Mòdul destí | Notes |
|---|---|---|
| `find_fonts`, `fit_font`, `render_char`, `render_all`, `show_sample` (01) | `ocr/dataset/render_fonts.py` | prep dades: renderitza 62 classes A–Z/0–9 a 64×64 negre-sobre-blanc, marge `MARGIN=6`. |
| `apply_geometric`, `apply_stroke`, `apply_contour`, **`apply_binarize` (Grup D)**, `augment`, `build_dataset`, `preview` (02) | `ocr/dataset/augment.py` | augmentation + binarització final (`adaptiveThreshold` MEAN_C/INV/31/15 a 64×64 → `resize 28×28 INTER_NEAREST`). Genera `train/`+`val/` + `class_to_idx.json`. |
| `class CharCNN` (03) | `ocr/model.py` | **separar de `main()`**. Arquitectura: 2× (Conv+BN+ReLU+MaxPool) → Dropout → FC → logits. Entrada 1×28×28, 62 classes. |
| `build_dataloaders`, `train_one_epoch`, `evaluate`, `plot_curves`, `main` (03) | `ocr/train.py` | entrenament + early stopping + corbes. Desa `models/char_cnn_best.pth` amb `model_state`, `n_classes`, `class_to_idx`, `val_acc`. |
| `load_model`, `predict_char`, `filter_by_confidence` (04) | `ocr/infer.py` | càrrega del checkpoint i inferència per caràcter. |
| `parse_gt`, `edit_distance`, `group_files`, `evaluate_group`, `compute_metrics`, `plot_confusion` (04) | `ocr/eval_real.py` | avaluació sobre caràcters reals (char/plate accuracy via distància d'edició + matriu de confusió). |

**API pública nova de l'OCR** (la que consumeix el pipeline):

```python
def load_model(model_path) -> tuple[nn.Module, dict, dict, torch.device]   # (model, class_to_idx, idx_to_class, device)
def predict(char_img_28x28, model, idx_to_class, device) -> tuple[str, float]   # (lletra, confiança)  ← afegir: accepta np.ndarray
def predict_char(model, img_path, device) -> tuple[int, float]   # firma ja existent a 04 (per a fitxers)
```

> **Adaptador necessari:** el pipeline treballa amb arrays a memòria, però `predict_char`
> de `04` rep una **ruta**. Cal una variant `predict()` que accepti l'`np.ndarray` 28×28 i
> reutilitzi el mateix `transform` (`Grayscale → Resize(28,28) → ToTensor → Normalize(0.5,0.5)`).

### Fase 3b — Reader (concepte de `CLAUDE.md`; implementar nou)

```python
def read_plate(chars: list[str]) -> str
```

Munta el string final concatenant els caràcters predits, en majúscules, i eliminant `?` i
símbols no alfanumèrics. **No aplica cap correcció contextual:** el dataset no segueix cap patró
(matrícules europees de 5–9 caràcters en ordre aleatori), així que no es pot inferir el tipus de
cap posició i corregir confusions a cegues només introduiria errors (veure ADR-007 revisat).
*No depèn de cap altra fase: és pura manipulació de strings.*

---

## 3. Punts d'entrada (`scripts/`) — substitueixen els bucles de notebook

| Script | Reemplaça | Què fa |
|---|---|---|
| `run_detector.py [img|dir] [--no-show]` | cel·les 8–10 del detector | Fase 1 sobre una imatge/directori; dibuixa boxes; resum de recall. |
| `run_segmenter.py [--metode --format-sortida --ocr --limit]` | `notebooks/fase_02/segmentador.py` | Fase 2 sobre `data/processed/`; genera chars 28×28 + visuals de depuració. |
| `build_ocr_dataset.py` | `01_render_fonts.py` + `02_augment.py` | encadena render → augment → `data/synthetic/{train,val}`. |
| `train_ocr.py [--epochs …]` | `03_train_cnn.py::main` | entrena la CNN → `models/`. |
| `evaluate.py` | `04_evaluate_real.py::main` + `diagnose_failures` | mètriques de detecció (recall) i d'OCR (char/plate accuracy ED, matriu de confusió). |
| `run_pipeline.py --input data/raw --output output/` | `notebooks/extra/04_pipeline.ipynb` | end-to-end: imatge → boxes → chars → lectura → `results.csv`. |

---

## 4. Contractes de dades entre fases (el més important)

```
imatge .jpg                      (BGR uint8, H×W×3)
   │  detector.detect()
   ▼
list[ROI]        retalls BGR uint8 d'1..N candidats per imatge (filosofia màxim recall)
   │  segmenter.segment(roi)     per a cada ROI
   ▼
list[char]       cada char: np.ndarray 28×28 uint8 BINARI {0,255}, BLANC sobre NEGRE
                 (llista BUIDA  ⇒  ROI rebutjada com a no-matrícula)
   │  ocr.predict(char, …)       per a cada char
   ▼
list[(lletra:str, conf:float)]
   │  reader.read_plate([lletra, ...])
   ▼
str              matrícula final (majúscules, sense espais)
```

**Invariants a respectar sempre:**

- El detector retorna `(x, y, w, h)` en **píxels de la imatge original** (no normalitzats).
- El caràcter de sortida del segmentador és **28×28, binari, blanc sobre negre** — exactament
  el que espera la CNN d'OCR.
- `segment()` retorna **`[]`** quan la ROI no supera el criteri de rebuig (5–9 caràcters +
  coherència de fila). Aquest és el mecanisme per descartar falsos positius del detector.
- Tot el que entra/surt d'una fase passa per `config` per als paràmetres; cap literal màgic.

---

## 5. `config.py` nou — contingut mínim

Es crea de zero. Agrupa per fase i evita els errors de l'antic config deprecat:

```
# Rutes
DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_CHARS_DIR, DATA_SYNTHETIC_DIR
MODELS_DIR, MODEL_CNN_PATH (= models/char_cnn_best.pth), OUTPUT_DIR
IMAGE_EXTENSIONS

# Fase 1 — Detector (de 01_morphologic_VJ.ipynb)
CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, GAUSS_KSIZE, GAUSS_SIGMA
CLOSE_KERNEL, OPEN_KERNEL
AREA_RATIO_MIN/MAX, ASPECT_RATIO_MIN/MAX, EXTENT_MIN, MIN_WIDTH, MIN_HEIGHT

# Fase 2 — Segmentador (de segmentador.py)
NORM_H, ADAPT_BLOCK, ADAPT_C, MORPH_CLOSE_KERNEL
DESKEW_MIN_CANDIDATS, DESKEW_ANGLE_MAX
H_CHAR_MIN_REL/MAX_REL, AR_MIN, AR_MAX, AREA_MIN_REL
H_MED_TOL, ROW_STD_H_MAX, ROW_STD_CY_MAX, WIDTH_OCC_MIN
N_CHARS_MIN = 5, N_CHARS_MAX = 9, IOU_OVERLAP
CANVAS_SIZE = 64, MARGIN = 6, OUTPUT_SIZE = 28
FORMAT_SORTIDA = "vj"          # "vj" (default, millor OCR) | "regla-or"

# Fase 3 — OCR (de notebooks/ocr/)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"   # 62 classes
OCR_INPUT_SIZE = 28
CONF_THRESHOLD = 0.5           # filtratge de baixa confiança
```

---

## 6. Ordre de migració recomanat

1. **`alpr/common/`** (io, geometry, annotations) — base sense dependències.
2. **`config.py`** nou — necessari per a tota la resta.
3. **OCR** — ja són `.py`: separar `CharCNN` (→ `model.py`), empaquetar `infer.py` amb
   `load_model`/`predict`. És el desbloqueig per provar el pipeline complet.
4. **Detector** — portar les 4 funcions pures + API `detect()/detect_boxes()/detect_debug()`.
5. **Segmentador** — reorganitzar `segmentador.py` en mòduls + API `segment()/save_chars()`.
6. **Reader** — implementar `read_plate()` (muntatge directe, sense correcció).
7. **`alpr/pipeline.py`** — encadenar i provar end-to-end sobre `data/raw`.
8. **`scripts/`** + `evaluate.py` — recrear bucles d'execució i mètriques.
9. **`alpr/viz.py`** — moure-hi totes les visualitzacions dels notebooks (opcional però net).

---

## 7. Decisions i fets del projecte a tenir presents (no silenciar-los)

1. **OCR = CNN PyTorch, NO HOG+SVM.** El pla SVM està abandonat. El model real és
   `models/char_cnn_best.pth` (62 classes, 28×28). Tota referència a SVM és deprecada.
2. **Caràcter = 28×28**, no 32×64. Unifica a 28×28 a tot el sistema.
3. **Segmentació = contorns** (`RETR_EXTERNAL`), no projecció vertical. La projecció és
   l'alternativa de comparació (`projection.py`), no el camí principal.
4. **Format de guardat del char = "vj"** (retall binari tight + `resize 28×28 INTER_AREA`,
   com el notebook `from_vj`). **Validat empíricament**: dona molt millor OCR que el llenç
   64×64 ("regla-or"). Veure `docs/segmentador-claude-code.md` §5. Mantén els dos formats
   seleccionables, però el default és "vj".
5. **Detector oficial = el morfològic** (`01_morphologic_VJ.ipynb`). Existeixen altres
   detectors al repo (Canny/NMS, rotated rects) — són referència, no el camí principal.
6. **Dependències d'OCR.** `torch`, `torchvision` i `Pillow` **NO** són a `requirements.txt`
   / `pyproject.toml` (només hi ha les de visió clàssica). En oficialitzar l'OCR com a part
   del paquet, **afegeix-les** i deixa-ho documentat.
7. **Rang de caràcters [5, 9]** (matrícules EU del dataset). `is_plausible_plate` ha de
   llegir-ho de `config`, no d'un literal.
8. **`data/` és fora de git** (gran). Els scripts han de crear els directoris de sortida si
   no existeixen.

---

### Resum d'una línia

Crea un paquet net `alpr/` des de zero (ignora `src/` sencer), amb subpaquets `detector/`
(← notebook morfològic), `segmenter/` (← `segmentador.py`, ja madur), `ocr/` (← scripts ja
`.py`, només empaquetar) i `reader/`; defineix els contractes de §4, centralitza els números
a un `config.py` nou, i treu tots els bucles i visuals als `scripts/`.
