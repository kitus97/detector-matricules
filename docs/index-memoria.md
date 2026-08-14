# Memòria tècnica — Sistema ALPR de lectura de matrícules

> Índex complet de la memòria final. L'estructura cobreix els quatre requisits obligatoris de
> l'enunciat: (i) explicació detallada del mètode, (ii) resultats amb ràtios i matriu de
> confusió, (iii) enumeració de funcions distingint les pròpies de les externes, (iv) annex
> amb tot el codi.

---

## 1. Introducció

- 1.1. Context i motivació del projecte
- 1.2. Objectiu: lectura automàtica de matrícules a partir de fotografies
- 1.3. Restriccions del projecte (sense models preentrenats; classificadors entrenats des de zero) i abast
- 1.4. Estructura de la memòria

## 2. Marc teòric

- 2.1. Preprocessament d'imatge: escala de grisos, suavitzat Gaussià, CLAHE
- 2.2. Detecció de vores i gradient: operador Sobel
- 2.3. Morfologia matemàtica: closing, opening, components connexos
- 2.4. Binarització: llindar global (Otsu) vs adaptatiu
- 2.5. Descriptors de característiques visuals: HOG (i per què es descarta com a detector principal)
- 2.6. Classificadors: SVM, k-NN i xarxes neuronals convolucionals (CNN) entrenades des de zero
- 2.7. Mètriques d'avaluació: recall, IoU, accuracy per caràcter i per matrícula, matriu de confusió

## 3. Disseny del sistema

- 3.1. Visió general del pipeline de 3 fases (detector → segmentador → OCR → reader)
- 3.2. Arquitectura modular i contractes entre fases (tipus de dades I/O)
- 3.3. Decisions d'arquitectura clau (síntesi dels ADRs)

## 4. Implementació per mòduls

### 4.1. Mòdul 1 — Detector de matrícules (visió clàssica morfològica)

- 4.1.1. Pipeline: Sobel vertical → Otsu → closing horitzontal → components connexos → filtre de forma
- 4.1.2. Filosofia de màxim recall i restricció de la cerca per projecció de files
- 4.1.3. Alternatives descartades (Harris/SIFT, HOG+SVM, Viola-Jones) i justificació

### 4.2. Mòdul 2 — Alineació i segmentació de caràcters

- 4.2.1. Reptes dels crops: il·luminació no uniforme, inclinació, falsos positius
- 4.2.2. Correcció d'inclinació (deskew) per regressió de centroides vs `minAreaRect`
- 4.2.3. Segmentació per contorns vs projecció vertical (comparació empírica: 79 vs 40 ROIs)
- 4.2.4. Binarització adaptativa, `RETR_EXTERNAL` i tractament de la 'I'
- 4.2.5. Criteri de rebuig de ROIs no-matrícula (rang [5,9] + coherència de fila)
- 4.2.6. Normalització de sortida a 28×28: format `vj` vs `regla-or` (comparació empírica)

### 4.3. Mòdul 3 — OCR (CNN entrenada des de zero)

- 4.3.1. Estratègia: dataset sintètic de fonts vs handwriting; el problema del domain gap
- 4.3.2. Generació de dades: render de fonts (`01_render_fonts.py`) i augmentation (`02_augment.py`)
- 4.3.3. Arquitectura de la CNN (`CharCNN`, 36 classes) i funció de pèrdua
- 4.3.4. Entrenament i monitorització

### 4.4. Mòdul 3b — Reader (muntatge final, sense correcció)

- 4.4.1. Muntatge del string final (neteja + concatenació de la predicció de l'OCR)
- 4.4.2. Per què NO s'aplica correcció contextual: el dataset no té format fix (matrícules
  europees de 5–9 caràcters en ordre aleatori) → no es pot inferir el tipus de cap posició

## 5. Experiments i resultats

- 5.1. Dataset i metodologia d'avaluació
- 5.2. Resultats del detector (recall centre del GT 98.1%, recall IoU≥0.5, candidats/imatge)
- 5.3. Resultats del segmentador (ROIs validades, caràcters generats, motius de rebuig)
- 5.4. Resultats de l'OCR (val sintètica 98.1%, char accuracy real, matriu de confusió)
- 5.5. Anàlisi del domain gap sintètic→real (+30.7 punts)
- 5.6. Resultats end-to-end (matrícules perfectes 22/79) i exemples qualitatius
- 5.7. Comparativa de tècniques (taula resum de les decisions validades empíricament)

## 6. Discussió

- 6.1. Interpretació dels resultats per fase i del coll d'ampolla (domain gap de l'OCR)
- 6.2. Limitacions conegudes (falsos positius del detector, plaques molt petites, format no fix)
- 6.3. Treball futur (afinar detector, reduir domain gap, filtre de confiança post-OCR)

## 7. Conclusions

## 8. Referències

*Bibliografia + tot el software de tercers utilitzat, correctament referenciat per evitar plagi.*

## Annex A — Enumeració de funcions

*Distingint clarament les funcions implementades per l'equip de les externes.*

## Annex B — Codi font complet

*Obligatori per enunciat.*
