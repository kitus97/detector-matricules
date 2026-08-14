# ADR-002 — Segmentació de caràcters per contorns

- **Estat:** ✅ Adoptat
- **Fase:** 2 — Segmentació
- **Mètrica oficial:** **79 ROIs acceptades** amb contorns vs **40** amb projecció vertical
  (mateixos 341 crops d'entrada)

---

## Context

La segona fase rep cada ROI candidata del detector i l'ha de partir en caràcters
individuals. Els crops presenten tres reptes: **il·luminació no uniforme** (reflexos, ombres),
**inclinació lleugera** (el detector no garanteix horitzontalitat) i **presència de falsos
positius** (no tots els crops contenen una matrícula). El nombre de caràcters per placa varia
(no hi ha format fix), per la qual cosa la tècnica no pot dependre de conèixer `n_chars` a
priori. Restricció dura: només processament clàssic, sense deep learning en aquest mòdul.

## Decisió

S'adopta la **detecció per contorns** com a mètode primari de segmentació:

```
adaptiveThreshold (MEAN_C, THRESH_BINARY_INV, blockSize=31, C=15)
  → MORPH_CLOSE (kernel vertical 1×3)
  → findContours (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
  → filtre geomètric (alçada i aspect ratio relatius a la mediana)
  → deduplicació IoU
```

Tots els llindars són **relatius a l'alçada de la placa**, mai en píxels absoluts. Abans de
detectar, el crop es **normalitza a 96 px d'alçada** (les plaques del dataset són de 18–46 px
i el `blockSize` fix de la binarització fallava sense normalitzar).

## Alternatives considerades

Les dues tècniques es van implementar i mesurar sobre els mateixos crops. Totes dues es
mantenen al repositori.

### Alternativa A — Projecció vertical (`char_segmenter.py`)

Binarització Otsu → amplificació 2.5× → histograma de projecció vertical (suma de píxels
blancs per columna) → les fronteres entre caràcters són els valls (mínims locals), suavitzats
amb mitjana mòbil de finestra 9.

- **Avantatges:** funciona bé quan `n_chars` es coneix a priori; robusta a inclinació lleugera;
  no genera dobles deteccions per forats interns (O, 0, D).
- **Limitacions:** requereix conèixer `n_chars` com a entrada i divideix la placa en
  exactament N parts (errònia amb matrícules de longitud variable); sensible a il·luminació no
  uniforme (Otsu global); els valls poden ser poc pronunciats amb caràcters molt junts.

### Alternativa B — Detecció de contorns ✅ **Adoptada**

- **Avantatges:** no requereix `n_chars` a priori (el nombre es descobreix de la detecció);
  més robusta a il·luminació no uniforme gràcies a l'adaptive threshold; la validació per rang
  és un post-filtre, no un paràmetre d'entrada.
- **Limitacions:** sensible a soroll que generi contorns de mida de caràcter; requereix
  deduplicació per IoU per a casos residuals.

## Justificació

**Mesura empírica decisiva:** sobre 341 crops, els contorns van acceptar **79 ROIs** vs **40**
de la projecció vertical — gairebé el doble. Les raons:

- **`RETR_EXTERNAL` en lloc de `RETR_TREE`:** `RETR_TREE` retorna també els contorns interiors
  (forats) de caràcters tancats com O, 0, D, A, B, generant dobles deteccions que descarten la
  placa per excés de caràcters. `RETR_EXTERNAL` ho elimina d'arrel.
- **`adaptiveThreshold` en lloc d'Otsu:** Otsu tria un llindar global únic; amb il·luminació no
  uniforme (ombra a un costat, reflex a l'altre) cap llindar global serveix. L'adaptiu compara
  cada píxel amb la mitjana del seu veïnat (blockSize=31), corregint gradients d'il·luminació.
- **`MORPH_CLOSE` vertical (1×3):** consolida traços verticals fins que la binarització
  fragmenta (especialment la 'I') sense fusionar caràcters adjacents.
- **Aspect ratio amb límit inferior 0.05:** necessari per capturar la 'I' i la '1', que tenen
  ratios de 0.05–0.12. Un llindar més estricte les descartaria silenciosament (risc detectat
  durant la crítica de disseny).

## Conseqüències

**Positives:**
- Captura matrícules de longitud variable sense conèixer-ne la longitud.
- 79 ROIs validades i 582 caràcters generats sobre el dataset.

**Negatives / riscos:**
- Soroll espacialment coherent (textures regulars del cotxe) pot passar els filtres
  geomètrics; el guardià definitiu és el criteri de rebuig (ADR-006) i, en el futur, el filtre
  de confiança de l'OCR.
- La 'I' continua sent el caràcter més difícil; el MORPH_CLOSE i l'aspect ratio 0.05 milloren
  la situació però no la resolen del tot.

## Alternatives no explorades

- **Projecció vertical amb N dinàmic** (`find_peaks` per descobrir N): dissenyada però no
  implementada; els contorns resolen el problema de forma més directa.
- **Segmentació per xarxa neuronal** (CRAFT, DBNet): més robusta però introdueix dependència de
  models preentrenats, fora de l'abast del projecte.

## Referències

- `docs/situacio-actual.md` §4; ADR-003 (deskew), ADR-005 (format de sortida), ADR-006 (rebuig)
