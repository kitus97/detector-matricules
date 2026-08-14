# Situació actual del projecte — ALPR (lectura de matrícules)

> Estat del sistema de lectura automàtica de matrícules amb **visió clàssica + una CNN
> petita d'OCR**, assumint el codi ja migrat de notebooks a un paquet Python (`.py`).
> Document pensat com a **material de suport per a la presentació**: cada secció acaba
> amb el "missatge clau" per a la slide. Totes les mètriques són mesures reals sobre el
> dataset del projecte (juny 2026).

---

## 1. Resum executiu

Sistema de **3 fases en cadena** que llegeix matrícules europees/espanyoles a partir de
fotos, **sense cap model de detecció preentrenat**: la detecció i la segmentació són
processament d'imatge clàssic (OpenCV), i el reconeixement de caràcters és una CNN
entrenada **des de zero** amb dades sintètiques generades per l'equip.

```
Foto del cotxe → [1] Detector → [2] Segmentador → [3] OCR (CNN) → [3b] Reader → "M5XSX"
```

| Fase | Tècnica | Mètrica titular |
|---|---|---|
| 1. Detecció | Morfologia + components connexos | **Recall 98.1%** (la matrícula entra entre 3.5 candidats/imatge) |
| 2. Segmentació | Contorns (`findContours` RETR_EXTERNAL) | 79/341 ROIs validades; rebuig automàtic de falsos positius |
| 3. OCR | CNN compacta (36 classes) | **98.1%** val sintètica · **87.5%** char accuracy real (alineat) |
| 3b. Lectura | Muntatge directe (sense correcció; el dataset no té patró) | la lectura final és la predicció de l'OCR, només netejada |

> **Missatge clau:** un pipeline 100% propi (sense detectors preentrenats) que captura el
> 98% de les matrícules i les llegeix amb una CNN entrenada amb fonts sintètiques.

---

## 2. Arquitectura del sistema (codi `.py`)

```
alpr/
├── detector/     Fase 1 — troba ROIs candidates a la imatge completa
├── segmenter/    Fase 2 — parteix cada ROI en caràcters individuals 28×28
├── ocr/          Fase 3 — classifica cada caràcter amb la CNN
├── reader/       Fase 3b — munta el string final (sense correcció)
├── common/       I/O, geometria (IoU/NMS), anotacions
└── pipeline.py   orquestrador end-to-end → output/results.csv
```

**Flux de dades i tipus entre fases:**

```
imatge BGR uint8
   │ detector.detect()
   ▼  1..N ROIs (retalls BGR) — filosofia de màxim recall
ROI
   │ segmenter.segment()   →  []  si es rebutja com a no-matrícula
   ▼  llista de caràcters 28×28 binaris, blanc sobre negre
chars
   │ ocr.predict()  per caràcter
   ▼  (lletra, confiança)
   │ reader.read_plate()
   ▼  "M5XSX"
```

> **Missatge clau:** disseny modular amb contractes nets; cada fase és substituïble i
> testejable per separat.

---

## 3. Fase 1 — Detector (visió clàssica morfològica)

**Pipeline:** gris → suavitzat Gaussià → CLAHE → **Sobel vertical** → binarització Otsu →
**closing horitzontal** + opening → components connexos → **filtre de forma** (àrea, aspect
ratio, extensió, mida).

**Idea central:** els caràcters d'una matrícula generen una **densitat anormalment alta de
vores verticals** en una regió acotada. El closing horitzontal fusiona aquestes vores en un
únic blob rectangular → la matrícula es converteix en una taca compacta detectable.

**Filosofia de disseny — màxim recall:** és preferible retornar diversos candidats (alguns
falsos positius) i **garantir que la matrícula real hi sigui sempre**, deixant que les fases
posteriors filtrin.

**Resultats (108 imatges amb ground truth):**

| Mètrica | Valor |
|---|---|
| Recall (centre del GT dins d'alguna caixa) | **98.1%** (106/108) |
| Recall (IoU ≥ 0.5 amb el GT) | 79.6% (86/108) |
| Candidats per imatge | mediana 3 · mitjana 3.5 · màx 11 |

> **Missatge clau:** 98% de recall amb només ~3.5 candidats per imatge — l'objectiu de la
> fase (no perdre cap matrícula) s'assoleix. El gap fins a IoU≥0.5 indica que la caixa de
> vegades no és perfectament ajustada, cosa que la Fase 2 corregeix amb el seu propi deskew
> i normalització.

---

## 4. Fase 2 — Segmentador (contorns + geometria)

**Pipeline per ROI:** normalització d'escala → deskew → binarització adaptativa + MORPH_CLOSE
→ `findContours(RETR_EXTERNAL)` → filtres geomètrics relatius a l'alçada de la placa →
selecció de la fila dominant + deduplicació IoU → **criteri de rebuig** → exportació 28×28.

**Decisions de disseny i el seu perquè:**

- **Contorns en lloc de projecció vertical** (decisió ADR-002). Mesura directa:
  **79 ROIs acceptades amb contorns vs 40 amb projecció** sobre els mateixos 341 crops. Els
  contorns no necessiten saber el nombre de caràcters a priori i són més robustos.
- **`RETR_EXTERNAL`** elimina el doble contorn (exterior + forat) de caràcters com O, 0, D, B.
- **Normalització d'escala a 96 px** abans de detectar: les matrícules del dataset són molt
  petites (18–46 px) i el `blockSize` fix de la binarització fallava sense normalitzar.
- **Deskew per regressió sobre els centroides** dels caràcters (no `minAreaRect` global, que
  donava sempre ≈0°).
- **Criteri de rebuig** (descarta la ROI sencera): menys de 5 o més de 9 caràcters, o fila
  incoherent (alçades/baseline/cobertura). És el mecanisme que **filtra els falsos positius
  del detector**.
- **Llindars sempre relatius a l'alçada de la placa**, mai en píxels absoluts.

**Resultats (341 crops d'entrada):** 79 acceptades, 262 rebutjades, **582 caràcters generats**.
Motius de rebuig: `pocs_caracters` 225, `baseline_inconsistent` 26, `massa_caracters` 8,
`alcades_inconsistents` 3.

### 4.1 L'experiment clau del format de sortida 28×28

Es van comparar **dos formats** de guardar el caràcter, i la mesura empírica amb la CNN va
ser decisiva (i contraintuïtiva):

| Format de sortida | Char accuracy (ED) | Plate accuracy | Char accuracy (posicional) |
|---|---|---|---|
| `regla-or` (llenç 64×64 + marge, replicant l'entrenament) | 37.7% | 10.1% | 58.1% |
| **`vj`** (retall binari + resize directe 28×28) ✅ | **67.5%** | **27.9%** | **87.5%** |

> **Missatge clau:** "valorar solucions" amb dades, no amb intuïcions. La teoria deia que
> calia replicar el llenç de l'entrenament; la mesura va demostrar que el format simple del
> notebook funciona molt millor (char accuracy gairebé el doble). Es va adoptar `vj`.

---

## 5. Fase 3 — OCR (CNN entrenada des de zero)

**Dataset 100% sintètic** generat per l'equip:

1. **Render de fonts** — 36 classes (A–Z, 0–9) renderitzades des de fonts `.ttf`/`.otf` a
   64×64, negre sobre blanc, amb marge.
2. **Augmentation** — degradació geomètrica (rotació, shear, escala), de traç (erosió/
   dilatació, soroll) i de contorn, i **binarització final** idèntica a la del segmentador
   real → 28×28 binari, blanc sobre negre, dividit en `train/`+`val/`.

**Model — `CharCNN`** (compacta): 2 blocs `Conv+BatchNorm+ReLU+MaxPool` (1→32→64) → Dropout →
`Linear(3136→256)` → Dropout → `Linear(256→36)`. Entrada 1×28×28.

> ⚠️ **Aclariment per a la presentació:** el sistema treballa amb **36 classes (A–Z, 0–9)**.
> Alguns comentaris del codi diuen "62 classes" per error (62 inclouria minúscules, que no
> es fan servir). El model entrenat té 36 classes.

**Resultats:**

| Mètrica | Valor |
|---|---|
| Validació sintètica (epoch 20) | **98.1%** |
| Char accuracy real, **posicional** (caràcters ben alineats) | **87.5%** (272 caràcters) |
| Char accuracy real via distància d'edició | 67.5% |
| Plate accuracy (matrícules perfectes) | referència 27.9% (22/79) · **pipeline alpr actual: 31.8% (28/88)** |
| Gap sintètic → real | +30.7% |

**Errors més freqüents** (de la matriu de confusió real): `1→7`, `8→B`, `6→5`, `2→Z`… Són
**domain gap** (la CNN entrena amb fonts sintètiques netes i infereix sobre fotos reals
degradades), no errors de segmentació.

> **Missatge clau:** la CNN aprèn perfectament les fonts (98%); el repte és el salt al
> domini real (fotos), un gap de ~30 punts típic quan s'entrena només amb dades sintètiques.

---

## 6. Fase 3b — Reader (muntatge directe, sense correcció)

Munta el string final concatenant els caràcters predits per l'OCR, en majúscules, i eliminant
`?` (baixa confiança) i símbols no alfanumèrics. **No aplica cap correcció contextual.**

**Per què cap correcció:** les matrícules d'aquest dataset **no segueixen cap patró**. Poden ser
de qualsevol país europeu (Sèrbia, Eslovènia, Turquia, Espanya…), amb A–Z + 0–9 en majúscules,
**5–9 caràcters en ordre totalment aleatori**. Cap posició té un tipus esperat (lletra o dígit),
de manera que no es pot saber la direcció de cap confusió (`S`↔`5`, `O`↔`0`…): corregir a cegues
només introduiria errors. Les confusions sistemàtiques de l'OCR són **domain gap** i s'han d'atacar
a la Fase 3 (augmentation / fine-tuning), no a la lectura. Veure ADR-007 (revisat).

> **Missatge clau:** sense un format de matrícula fix, la lectura es limita a muntar la millor
> predicció de l'OCR; el reader no pot (ni ha de) corregir confusions a cegues.

---

## 7. Resultats globals i lectura end-to-end

Exemples reals del pipeline complet (segmentador `vj` + CNN):

| Matrícula (GT) | Lectura del sistema | Comentari |
|---|---|---|
| `M5XSX` | `M5XSX` ✅ (conf 1.00) | perfecta; el guió "-" es rebutja correctament |
| `WA56660` | `WAS666OX` | gairebé exacta (S↔5, O↔0, +1 caràcter de la banda UE) |
| `BS47040` | `…BS47O4O` | placa correcta dins (O↔0) + soroll de la banda UE |

**Resum quantitatiu de tot el sistema:**

| Fase | Mètrica clau | Valor |
|---|---|---|
| Detector | Recall (centre GT) | 98.1% |
| Segmentador | ROIs validades / total crops | 79 / 341 |
| OCR (alineat) | Char accuracy posicional | 87.5% |
| Sistema | Matrícules perfectes | referència 22/79 · **pipeline alpr actual: 28/88** (`python -m alpr.ocr.eval_real`) |

---

## 8. Decisions clau i alternatives descartades (la "història" per a la presentació)

| Decisió | Alternativa descartada | Per què |
|---|---|---|
| Detector morfològic (Sobel-v) | Corner detection, HOG+SVM, plate_detector | Millors resultats i més simple |
| Segmentació per contorns | Projecció vertical (histograma) | 79 vs 40 ROIs; no cal saber el nombre de caràcters |
| `RETR_EXTERNAL` | `RETR_TREE` | Evita el doble contorn de O/0/D/B |
| Deskew per regressió de centroides | `minAreaRect` global | El global donava angle ≈0° sempre |
| Format de char `vj` | Format `regla-or` (llenç 64×64) | 67% vs 38% char accuracy (mesura empírica) |
| OCR: CNN PyTorch | HOG + SVM | Pla SVM abandonat; la CNN dona millors resultats |
| Normalització a 96 px | Processar el crop a mida original | Les plaques de 18–46 px trencaven la binarització |

> **Missatge clau:** cada decisió està **justificada amb una mesura o un problema concret**,
> documentada a `docs/adr/` i a l'auditoria del segmentador.

---

## 9. Limitacions conegudes i treball futur

- **Falsos positius del detector** (banda blava de la UE, text del concessionari sota la
  placa): generen caràcters extra que la confiança de la CNN no sempre filtra. És sobretot
  responsabilitat de la Fase 1.
- **Domain gap de l'OCR** (+30 punts): la CNN entrena amb fonts netes. Millorable amb
  augmentation més realista o fine-tuning amb caràcters reals etiquetats.
- **Plaques molt petites** (18 px): poca informació de partida; segmentació amb soroll tot i
  la normalització.
- **Sense format fix de matrícula:** el dataset no segueix cap patró (5–9 caràcters EU en ordre
  aleatori), així que el reader no pot validar ni corregir per posició; es limita a muntar la
  predicció de l'OCR. Recuperar les confusions sistemàtiques és feina de la Fase 3 (domain gap).

**Línies de millora prioritàries:** (1) afinar el detector per reduir FP en origen;
(2) reduir el domain gap de l'OCR; (3) filtre de qualitat de ROI abans de segmentar.

---

## 10. Quadre de mètriques per a slides (còpia ràpida)

```
DETECTOR        Recall (centre GT) ........ 98.1%  (106/108)
                Recall (IoU>=0.5) ......... 79.6%  (86/108)
                Candidats/imatge .......... 3.5 de mitjana

SEGMENTADOR     ROIs acceptades ........... 79 / 341
                Caràcters generats ........ 582
                Contorns vs projecció ..... 79 vs 40 ROIs

OCR (CNN)       Classes ................... 36 (A-Z, 0-9)
                Val sintètica ............. 98.1%
                Char acc real (alineat) ... 87.5%
                Char acc real (ED) ........ 67.5%
                Matrícules perfectes ...... 28 / 88  (31.8%)  [pipeline alpr; char acc ED 73.7%]
                Gap sintètic→real ......... +30.7%

FORMAT 28x28    vj vs regla-or (char ED) .. 67.5% vs 37.7%
```

> **Frase de tancament:** un sistema ALPR complet i propi on cada fase compleix el seu
> objectiu (detectar amb màxim recall, segmentar i filtrar, reconèixer amb una CNN pròpia),
> amb totes les decisions validades empíricament i el coll d'ampolla actual ben identificat:
> el salt del domini sintètic al real a l'OCR.
