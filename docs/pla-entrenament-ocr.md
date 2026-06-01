# Pla d'entrenament OCR — CNN amb fonts sintètiques

## Context

Entrenament d'un classificador de caràcters alfanumèrics (A–Z, 0–9 → **62 classes**) per al mòdul OCR del pipeline de detecció de matrícules. Les dades d'entrenament són **sintètiques**, generades a partir de fonts tipogràfiques i augmentades per imitar els artefactes del segmentador real. No es fa servir el Ground Truth del dataset per a l'entrenament (reservat per a validació).

---

## Pas 1 — Recollir les fonts

Descarrega un conjunt de fonts variades que cobreixi:

- **Sans-serif estàndard** (Arial, DejaVu Sans, Liberation Sans): forma base dels caràcters
- **Monoespaciades** (Courier, DejaVu Mono): caràcters més quadrats, similars a moltes matrícules
- **Bold / condensades**: simula matrícules amb traç gruixut
- **FE-Schrift o DIN 1451** (si estan disponibles): fonts oficials de matrícules europees

**Objectiu mínim: 10–20 fonts.** Més fonts = més variabilitat = millor generalització.

Guarda-les totes a `resources/fonts/`.

---

## Pas 2 — Renderitzar els caràcters base

Per cada font i per cada caràcter del conjunt `A-Z 0-9`, renderitza la lletra en **negre sobre fons blanc** a una mida generosa (64×64 px) amb `PIL.ImageFont` + `PIL.ImageDraw`.

> **Resultat esperat:** `62 caràcters × 20 fonts = 1.240 imatges base netes`

---

## Pas 3 — Decidir la mida de sortida

El segmentador ja guarda els caràcters a **28×28 px**. Totes les imatges d'entrenament han d'estar en aquest mateix format per eliminar qualsevol discrepància de resolució entre entrenament i inferència.

Totes les imatges base es redimensionen a 28×28 al final del pipeline d'augmentation.

---

## Pas 4 — Dissenyar el pipeline d'augmentation

Cada transformació s'aplica amb una probabilitat (no sempre totes). L'ordre dels grups és important: primer la geometria, després el traç, després la binarització final.

### Grup A — Degradació geomètrica
> Imita el deskew imperfecte i la segmentació

| Transformació | Rang |
|---|---|
| Rotació aleatòria | ±8° |
| Shear horitzontal | ±0.1 |
| Desplaçament aleatori | ±10% en x i y |
| Escala aleatòria | 0.85 – 1.15 |

### Grup B — Degradació de traç
> Imita la binarització adaptive i la variació d'il·luminació

| Transformació | Paràmetres |
|---|---|
| Erosió morfològica | kernel 1×1 o 2×2 (traç més prim) |
| Dilatació morfològica | kernel 1×1 o 2×2 (traç més gruixut) |
| Soroll salt-and-pepper | 1–3% de píxels afectats |

### Grup C — Degradació de contorn
> Imita el tall de bounding box del segmentador

| Transformació | Rang |
|---|---|
| Crop lleuger d'un costat aleatori | 0–3 px |
| Descentrat del caràcter al llenç | ±3 px en x i y |

### Grup D — Binarització final ⚠️

Aplica el **mateix `adaptiveThreshold(blockSize=31, C=15)`** que usa el segmentador real. Aquest és el pas més crític de tot el pipeline d'augmentation: la textura de les vores ha de ser idèntica a la de les imatges reals que el model veurà durant la inferència.

---

## Pas 5 — Decidir el volum de dades

| Concepte | Valor |
|---|---|
| Mostres per classe | 500 |
| Classes | 62 |
| Total imatges | 31.000 |
| Split entrenament | 80% → 24.800 |
| Split validació | 20% → 6.200 |

Les classes han d'estar **perfectament balancejades** (mateix N per classe). Si el model fa overfitting augmenta el volum; si entrena massa lent redueix-lo.

---

## Pas 6 — Estructurar el dataset en disc

Format compatible amb `torchvision.ImageFolder`, on cada subcarpeta és el nom de la classe:

```
data/synthetic/
├── train/
│   ├── A/
│   │   ├── img_0001.png
│   │   ├── img_0002.png
│   │   └── ...
│   ├── B/
│   ├── 0/
│   └── ... (62 carpetes)
└── val/
    ├── A/
    └── ...
```

> `ImageFolder` llegeix l'estructura automàticament i assigna un índex numèric a cada classe. Assegura't de guardar el `class_to_idx` (el mapa classe→índex) per poder descodificar les prediccions durant la inferència.

---

## Pas 7 — Definir l'arquitectura del CNN

Arquitectura compacta adequada per a imatges binàries 1×28×28:

```
Input       1 × 28 × 28
Conv2d(1→32, k=3) + BatchNorm + ReLU + MaxPool(2)    →  32 × 13 × 13
Conv2d(32→64, k=3) + BatchNorm + ReLU + MaxPool(2)   →  64 × 5 × 5
Dropout(0.4)
Flatten                                               →  1.600
Dense(256) + ReLU
Dropout(0.3)
Dense(62) + Softmax
```

**Hiperparàmetres de partida:**

| Hiperparàmetre | Valor |
|---|---|
| Optimitzador | Adam |
| Learning rate | 1e-3 |
| Loss | CrossEntropyLoss |
| Epochs | 20–30 (amb early stopping) |
| Batch size | 64 o 128 |

La funció de pèrdua és la Cross-Entropy estàndard per classificació multiclasse:

$$\mathcal{L} = -\sum_{c=1}^{62} y_c \log(\hat{y}_c)$$

---

## Pas 8 — Monitoritzar l'entrenament

Guarda les corbes de loss i accuracy (train vs val) a cada epoch per detectar:

- **Overfitting**: val loss puja mentre train loss baixa → augmenta el dropout o el volum d'augmentation
- **Underfitting**: tots dos alts → model més gran o menys regularització
- **Bon entrenament**: les dues corbes convergeixen juntes cap a un valor baix

Guarda el checkpoint del model amb millor `val_accuracy` (no el de l'última epoch).

---

## Pas 9 — Avaluar sobre caràcters REALS

Extrau manualment 5–10 matrícules del dataset on el segmentador hagi funcionat bé, etiqueta els caràcters a mà usant el GT disponible i mesura:

| Mètrica | Descripció |
|---|---|
| **Char Accuracy** | % de caràcters individuals correctes (la principal) |
| **Plate Accuracy** | % de matrícules completes sense cap error |
| **Matriu de confusió** | Identifica parells problemàtics (0/O, 1/I, 5/S, 8/B) |

> ⚠️ Si l'accuracy real és ≥ 10 punts per sota de la de validació sintètica, el domain gap és significatiu. Revisa el **Pas 4** — probablement l'augmentation no captura prou bé els artefactes reals.

---

## Pas 10 — Integrar amb el pipeline

El model entrenat s'exposa com un mòdul:

```python
def predict(char_img_28x28: np.ndarray) -> tuple[str, float]:
    """
    Retorna (lletra_predita, score_confiança).
    score és la probabilitat màxima del Softmax ∈ [0, 1].
    """
```

El `score` de confiança serà el que s'usarà al **filtre de confiança post-OCR** (documentat com a TODO a l'ADR-002) per descartar caràcters amb predicció poc fiable.

---

## Resum del flux

```
Fonts (10–20)
    └─► Pas 1-2: Renderitzar 62 classes (1×64×64, negre/blanc)
            └─► Pas 4: Augmentation (geomètrica → traç → contorn → binaritzar)
                    └─► Pas 3: Redimensionar a 28×28
                            └─► Pas 5-6: Dataset estructurat (31.000 imatges, balancejat)
                                    └─► Pas 7-8: Entrenar CNN (62 classes, CrossEntropy)
                                            └─► Pas 9: Validar amb caràcters REALS
                                                    └─► Pas 10: Integrar al pipeline OCR
```

---

## Decisions pendents

- [ ] Confirmar que FE-Schrift / DIN 1451 estan disponibles lliurement
- [ ] Decidir si es genera el dataset una sola vegada (estàtic) o on-the-fly durant l'entrenament
- [ ] Definir el llindar de confiança mínim per al filtre post-OCR (Pas 10)
- [ ] Decidir si es fa fine-tuning amb GT real si el domain gap (Pas 9) és massa gran
