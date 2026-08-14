# ADR-004 — OCR amb CNN entrenada amb fonts sintètiques

- **Estat:** ✅ Adoptat
- **Fase:** 3 — OCR
- **Mètrica oficial:** **98.1%** val sintètica · **87.5%** char accuracy real (posicional)

---

## Context

La tercera fase classifica cada caràcter segmentat (imatge 28×28 binària) en una de les
**36 classes** (A–Z, 0–9). El projecte permet deep learning sempre que sigui **entrenat per
l'equip** (no preentrenat). No es disposava inicialment de caràcters reals etiquetats, però es
poden extreure del propi dataset per a validació. El repte central és el **domain gap**: el
model s'ha d'entrenar amb dades que s'assemblin als caràcters que produeix el segmentador real.

## Decisió

S'adopta una **CNN compacta entrenada des de zero amb un dataset 100% sintètic basat en
fonts tipogràfiques**, amb augmentation que imita els artefactes del pipeline propi.

**Pipeline de dades** (`01_render_fonts.py` + `02_augment.py`):
1. **Render de fonts** — 36 classes renderitzades des de fonts `.ttf`/`.otf` a 64×64, negre
   sobre blanc, amb marge.
2. **Augmentation** — degradació geomètrica (rotació ±5–8°, shear, escala), de traç
   (erosió/dilatació, soroll sal-i-pebre i gaussià), tall parcial de vores, i **binarització
   final idèntica a la del segmentador real** → 28×28 binari, blanc sobre negre.

**Model — `CharCNN`:** 2 blocs `Conv+BatchNorm+ReLU+MaxPool` (1→32→64) → Dropout →
`Linear(3136→256)` → Dropout → `Linear(256→36)`. Entrada 1×28×28. Pèrdua Cross-Entropy:

$$\mathcal{L} = -\sum_{c=1}^{36} y_c \log(\hat{y}_c)$$

## Alternatives considerades

| Alternativa | Per què es va descartar |
|---|---|
| **Dataset de handwriting (EMNIST/MNIST)** | Domini visualment molt diferent: el handwriting aprèn variabilitat de traç humà que **no existeix** a les matrícules (fonts tipogràfiques de gruix uniforme). Entrenar amb handwriting i inferir sobre matrícules és un domain gap clàssic que enfonsa l'accuracy real. |
| **HOG + SVM** | Plantejat com a classificador clàssic alternatiu; abandonat perquè la CNN dona millors resultats sobre 36 classes amb imatges 28×28 binàries. |
| **Arquitectures grans (ResNet, etc.)** | Sobredimensionades per a 36 classes 28×28; més propenses a overfitting amb dades sintètiques i innecessàries (la CNN compacta entrena en minuts). |

## Justificació

La robustesa no s'aconsegueix amb "imperfeccions qualssevol", sinó amb **les imperfeccions que
el segmentador propi produeix**: fragments per binarització, soroll de l'adaptive threshold,
inclinació residual, traços tallats. Per això:

- **Fonts sintètiques** apropen el domini d'entrenament al real (fonts tipogràfiques de gruix
  uniforme), a diferència del handwriting.
- **Augmentation que replica el pipeline** (mateixa binarització que el segmentador) minimitza
  el domain gap: com més s'assembli la generació al pipeline real, menor és el gap.
- **Validació amb caràcters reals etiquetats**, no només sintètics: l'accuracy sobre sintètic
  és enganyosa perquè el model ha vist exactament aquesta distribució.

## Conseqüències

**Positives:**
- La CNN aprèn les fonts gairebé perfectament (98.1% val sintètica).
- Sistema 100% propi, sense models preentrenats.
- El `score` de confiança (màxim del Softmax) queda disponible per a un futur filtre de
  confiança post-OCR.

**Negatives / riscos:**
- **Domain gap de +30.7 punts** (98.1% sintètic → 67.5% char accuracy real via distància
  d'edició). Els errors freqüents (`1→7`, `8→B`, `6→5`, `2→Z`) són domain gap, no errors de
  segmentació: la CNN entrena amb fonts netes i infereix sobre fotos reals degradades.
- Mitigable amb augmentation més realista o fine-tuning amb caràcters reals etiquetats (treball
  futur).

## Nota de coherència

El sistema oficial té **36 classes** (A–Z, 0–9). Algunes referències del codi i de la fase de
disseny parlen de "62 classes" (que inclourien minúscules); són obsoletes. El model entrenat
(`models/char_cnn_best.pth`) té 36 sortides.

## Referències

- `docs/situacio-actual.md` §5; `notebooks/ocr/01_render_fonts.py`, `02_augment.py`
- ADR-005 (format de sortida 28×28), ADR-007 (lectura final sense correcció)
