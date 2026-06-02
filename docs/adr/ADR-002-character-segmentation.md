# ADR-002: Segmentació de caràcters de matrícula

| Camp | Valor |
|---|---|
| **Estat** | Acceptat |
| **Data** | 2026-05-28 |
| **Autors** | Equip detector-matricules |
| **Fitxers afectats** | `char_segmenter.py`, `notebooks/02_character_segmentation_from_vj.ipynb` |

---

## Context

Un cop el detector morfològic (Viola-Jones) ha retallat les regions candidates de la imatge original i les ha guardat com a `{stem}_box{n}.png` a `data/processed/`, cal extreure els caràcters individuals de cada crop per passar-los a l'OCR.

Els crops presenten tres reptes principals:

- **Il·luminació no uniforme**: reflexos, ombres del para-xocs o variació de llum natural fan que un llindar global falli en zones de la mateixa imatge.
- **Inclinació lleugera**: el detector VJ no garanteix que la placa estigui perfectament horitzontal.
- **Presència de falsos positius**: no tots els crops contenen una matrícula real; alguns són regions de textura que el detector ha confós amb una placa.

El nombre esperat de caràcters per matrícula en el dataset és entre **5 i 8**.

---

## Decisió 1: correcció d'inclinació prèvia (deskew)

### Opció descartada — `minAreaRect` global

Aplicar `cv2.minAreaRect` sobre tots els píxels de primer pla del crop. Es va implementar inicialment però es va descartar: la núvol global de punts (que inclou el fons binaritzat, vores de la placa i soroll) forma una forma aproximadament rectangular alineada amb els eixos de la imatge, de manera que l'angle resultant és gairebé sempre ≈ 0° independentment de la inclinació real.

### Decisió adoptada — esquema de dues passades amb regressió sobre centroides

**Passada 1 (tolerant):** Es binaritza el crop amb `adaptiveThreshold` i es detecten contorns amb filtres laxos (alçada entre el 70–130% de la mediana, aspect ratio 0,05–1,2). L'objectiu d'aquesta passada no és segmentar perfectament sinó localitzar prou caràcters candidats per estimar l'angle.

**Estimació de l'angle:** Es calcula el centroide de cada bbox candidat `(cx, cy)` i s'ajusta una recta per regressió lineal:

$$m, b = \text{polyfit}(cx_i, cy_i, 1)$$

L'angle de correcció és:

$$\alpha = \arctan(m) \cdot \frac{180}{\pi}$$

Els caràcters d'una matrícula estan alineats horitzontalment per disseny, de manera que la línia de centroides revela directament la inclinació de la placa. Si `|α| > 15°` o hi ha menys de 3 candidats, no es rota (senyal insuficient o detecció errònia).

**Rotació:** `cv2.warpAffine` al voltant del centre de la imatge amb `INTER_CUBIC` i `BORDER_REPLICATE` per evitar franges negres als cantons.

**Passada 2 (estricta):** Sobre la placa ja alineada, es repeteix la detecció amb filtres estrictes. Ara l'aspect ratio dels caràcters és geomètricament real i els filtres es poden aplicar amb confiança.

**Justificació del disseny en dues passades:** Una inclinació de 5–10° no impedeix detectar els caràcters per a l'estimació de l'angle. Per estimar un pendent no es necessiten tots els caràcters perfectes; amb 4–5 centroides ben repartits la recta és fiable. En canvi, per a la segmentació final (passada 2) les condicions han de ser ideals.

---

## Decisió 2: tècnica de segmentació

Es van avaluar dues tècniques. Totes dues es mantenen al repositori com a aproximacions alternatives.

### Alternativa A — Projecció vertical (`char_segmenter.py`)

**Descripció:** Es binaritza amb Otsu, s'amplifica la imatge 2,5× i es calcula l'histograma de projecció vertical (suma de píxels blancs per columna). Les fronteres entre caràcters corresponen als valleys (mínims locals) d'aquest histograma. Una mitjana mòbil de finestra 9 suavitza el soroll del histograma. Es busca el mínims local en una finestra de ±5% de l'amplada al voltant de cada frontera esperada.

**Avantatges:**
- Funciona bé quan el número de caràcters és conegut a priori.
- Robusta davant inclinació lleugera.
- No genera dobles deteccions per forats interns (O, 0, D).

**Limitacions:**
- Requereix conèixer `n_chars` com a entrada: el segmentador divideix la placa exactament en N parts, que pot produir segments erronis quan les matrícules tienen diferent número de caràcters.
- Sensible a il·luminació no uniforme (Otsu global).
- Els valleys del histograma poden ser poc pronunciats quan caràcters estan molt junts.

### Alternativa B — Detecció de contorns (`02_character_segmentation_from_vj.ipynb`) ✓ **Adoptada**

**Descripció:** `cv2.adaptiveThreshold` (MEAN_C, THRESH_BINARY_INV, blockSize=31, C=15) seguida de `cv2.findContours` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) i filtratge per alçada mediana i aspect ratio.

**Per què `adaptiveThreshold` en lloc d'Otsu:** Otsu tria un únic llindar global per a tota la imatge. Amb il·luminació no uniforme (ombra a un costat, reflexos a l'altre), el llindar bo per a la zona il·luminada no serveix per a la zona fosca. `adaptiveThreshold` compara cada píxel amb la mitjana d'un veïnat local (blockSize=31), corregint els gradients d'il·luminació independentment.

**Per què `RETR_EXTERNAL` en lloc de `RETR_TREE`:** `RETR_TREE` retorna tant els contorns exteriors com els interiors (forats) de cada caràcter tancat. Els caràcters com 'O', '0', 'D', 'A', 'B' generen dos contorns que tots dos passen el filtre geomètric, causant dobles deteccions i matrícules descartades per excés de caràcters o dígits repetits.

**Filtre geomètric (passada estricta):**
- Alçada entre el 85% i 115% de la mediana d'alçades (els caràcters reals d'una matrícula tenen alçades molt uniformes).
- Aspect ratio entre 0,05 i 0,95 (el límit inferior de 0,05 és necessari per capturar la 'I', que pot tenir ratios de 0,05–0,12).

**Tractament morfològic pre-segmentació:** `cv2.MORPH_CLOSE` amb kernel vertical (1×3) just després de l'`adaptiveThreshold`. Consolida traços verticals fins fragmentats per la binarització (especialment la 'I') sense fusionar caràcters adjacents.

**Avantatges sobre la projecció vertical:**
- No requereix conèixer `n_chars` a priori: el nombre es descobreix a partir de la detecció.
- Més robusta davant il·luminació no uniforme gràcies a l'adaptive threshold.
- Validació per rang [5, 8] com a post-filtre, no com a paràmetre d'entrada.

**Limitacions:**
- Sensible a soroll que generi contorns del tamany d'un caràcter.
- Requereix un pas addicional de deduplicació per IoU per casos residuals de dobles deteccions.

---

## Decisió 3: validació de qualitat del crop

### Rang de caràcters vàlid

Un crop s'accepta únicament si el nombre de caràcters detectats es troba a **[5, 8]**. Fora d'aquest rang, el crop es considera un fals positiu del detector VJ o una segmentació errònia i es descarta sense guardar res.

Constrants: `N_CHARS_MIN = 5`, `N_CHARS_MAX = 8`.

### Coherència geomètrica (`is_plausible_plate`)

A més del recompte, s'apliquen tres validacions sobre els bboxes detectats:

| Comprovació | Llindar | Justificació |
|---|---|---|
| Desviació estàndard d'alçades | ≤ 20% de la mediana | Caràcters reals tenen alçades molt uniformes |
| Desviació estàndard de `cy` | ≤ 25% de la mediana | Caràcters reals comparteixen línia base |
| Amplada total ocupada | ≥ 30% de l'amplada del crop | Una matrícula real ocupa la major part del crop |

Aquests filtres descarten crops de soroll que casualment generin entre 5 i 8 blobs del tamany d'un caràcter però sense la coherència espacial d'una matrícula real.

### Post-filtrat per confiança d'OCR (pendent)

Quan l'OCR estigui implementat, s'afegirà un filtre addicional basat en la confiança per caràcter. Com que les matrícules del dataset no segueixen un format fix (no es pot validar per patró NNNN-LLL), el filtre es basarà exclusivament en el score de confiança. Si després del filtrat queden menys de `N_CHARS_MIN` caràcters, la matrícula es descarta. Documentat a `TODO` al notebook.

---

## Conseqüències

**Positives:**
- La correcció d'inclinació per regressió sobre centroides és robust perquè es basa en el senyal rellevant (alineació dels caràcters), no en la forma global del crop.
- La detecció per contorns no requereix conèixer el número de caràcters a priori, cosa que facilita el tractament de matrícules de longitud variable.
- `RETR_EXTERNAL` elimina d'arrel el problema de dobles contorns en caràcters amb forats, sense necessitat de lògica addicional de deduplicació (tot i que es manté `remove_overlapping_bboxes` com a xarxa de seguretat).

**Negatives / riscos:**
- L'esquema de dues passades aplica la binarització dues vegades per crop, incrementant el temps de processament. Acceptable per a un dataset acadèmic; caldria optimitzar per a producció.
- Els falsos positius del detector VJ que generin soroll coherent espacialment (textures regulars del cotxe) poden passar els filtres geomètrics. El filtre de confiança d'OCR serà el guardià definitiu.
- La 'I' continua sent el caràcter més difícil de detectar. El MORPH_CLOSE vertical i el límit d'aspect ratio de 0,05 milloren la situació però no la resolen completament en tots els casos.

---

## Alternatives no explorades

- **Projecció vertical amb N dinàmic**: substituir la projecció de N fix per detecció de pics (`find_peaks`) per descobrir N automàticament. Es va dissenyar el canvi però no s'ha implementat, ja que la tècnica de contorns resol el problema de forma més directa.
- **Segmentació per xarxa neuronal** (ex. CRAFT, DBNet): produiria deteccions molt més robustes però introduiria una dependència de models preentrenats i és fora de l'abast del projecte, que requereix implementació pròpia.
- **Hough Transform per estimar l'angle**: alternativa a la regressió sobre centroides, més robusta davant outliers però més complexa d'implementar i innecessàriament pesada per a inclinacions lleugeres (< 15°).
