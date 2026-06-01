# Auditoria crítica del segmentador de caràcters (`02_character_v2.ipynb`)

| Camp | Valor |
|---|---|
| **Data** | 2026-05-29 |
| **Fitxer auditat** | `notebooks/fase_02/02_character_v2.ipynb` |
| **Evidència** | `models/eval_real_results.csv` (90 grups, 654 files de caràcter), 341 crops `_box` a `notebooks/data/processed/`, recompute read-only del propi algorisme de v2 |
| **Abast** | Només anàlisi i documentació. No s'ha modificat cap codi. |

> ⚠️ **Avís transversal sobre l'evidència (llegir abans de tot).** El CSV `eval_real_results.csv`
> **NO s'ha generat amb `02_character_v2.ipynb`**. Veure §1.5. El CSV prové dels caràcters de
> `notebooks/data/chars/` (nomenclatura `_box`), produïts per un segmentador germà executat amb
> `cwd=notebooks/`. En reexecutar l'algorisme **exacte** de v2 sobre els mateixos 341 crops obtinc
> **74 crops acceptats**, no els **90 grups** del CSV. Per tant les xifres de §2 són la millor
> aproximació disponible a la tècnica de contorns de dues passades, però **v2 mai no s'ha avaluat
> de punta a punta sobre cap dataset**. Aquesta és la troballa més important de l'auditoria.

---

## Secció 1 — Inventari real del que fa el codi avui

**`detect_char_bboxes(thresh, strict)` (cel·la `cell-03-functions`).** Fa
`findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` sobre la imatge binaritzada, calcula la mediana
d'alçades (ignorant blobs amb `h ≤ 10 px`) i filtra per alçada i aspect ratio. En mode tolerant
(`strict=False`, passada 1) accepta alçada 70–130% de la mediana i AR 0.05–1.20; en mode estricte
(`strict=True`, passada 2) accepta alçada 85–115% i AR 0.05–0.95. Retorna els bboxes ordenats per `x`.

**`estimate_skew_angle(bboxes)`.** Ajusta una recta per mínims quadrats (`np.polyfit`) als centroides
dels bboxes i retorna `arctan(pendent)` en graus. Retorna **0.0** en dos casos: si hi ha **menys de 3
bboxes** (senyal insuficient) o si **`|angle| > 15°`** (`ALIGN_ANGLE_MAX`, es considera detecció
errònia). Mesura empírica sobre 341 crops: **226/341 (66%)** tenen <3 candidats a la passada 1 i per
tant **mai no s'intenta corregir la inclinació**; només 108 ho intenten i 7 queden bloquejats pel
límit de 15°. L'angle medià és 0.00° i només 16 crops superen els 5°.

**`process_crop(crop_bgr)`.** Pipeline de dues passades: (1) binaritza amb
`adaptiveThreshold(MEAN_C, BINARY_INV, blockSize=31, C=15)` i estima l'angle amb filtres tolerants;
(2) rota amb `warpAffine(INTER_CUBIC, BORDER_REPLICATE)`; (3) re-binaritza i detecta amb filtres
estrictes; (4) accepta el crop només si el recompte cau a **[5, 10]** (`N_CHARS_MIN=5`,
`N_CHARS_MAX=10`); (5) retalla cada caràcter amb `thresh2[y:y+h, x:x+w]` (**sense marge**) i el
redimensiona directament a **28×28** (`cell-03`, línia `cv2.resize(thresh2[y:y+h, x:x+w], (28,28))`).

**Constants i si tenen sentit per al dataset.** `N_CHARS_MAX=10` és més permissiu que el [5,8] de
l'ADR-002: les matrícules reals del dataset tenen 7 caràcters (p. ex. `eu10`=`WA56660`,
`test_001`=`PP587AO`, `test_065`=`4BO4979`), per tant el marge fins a 10 **admet 3 falsos positius
per matrícula sense rebutjar el crop** — contribueix directament al problema de §2.1. `AR_floor=0.05`
és raonable: el blob estret real més extrem mesurat té AR=0.071 (§2.2). `ALIGN_ANGLE_MAX=15°` rarament
és rellevant perquè el deskew gairebé mai s'activa (§2.4).

**1.5 — Desconnexió de dades i versions (crític).** Tres datasets conviuen:

| Ruta | Contingut | Origen | Estat |
|---|---|---|---|
| `data/processed/` | 519 `_cand*.jpg` | detector nou `main_3.py` | **sense segmentar** (`data/chars/` és buit) |
| `notebooks/data/processed/` | 341 `_box*.png` | detector VJ antic | font dels chars del CSV |
| `notebooks/data/chars/` | 654 `_box*_char*.png` | segmentador germà (cwd=notebooks/) | font del CSV |

- `_glob_crops` de v2 fa `directory.glob('*_box*.png')` (cel·la `cell-01`, línia 95). Sobre
  `data/processed/` (que conté `_cand*.jpg`) **retorna 0 fitxers**. Tot i que `parse_crop_filename`
  suporta `_cand`, el glob mai els arriba a llegir, i a més filtra `.png` però els crops nous són
  `.jpg`. **v2 no pot processar la sortida del detector actual sense tocar el glob i l'extensió.**
- v2 escriu a `data/chars/` (buit). El CSV prové de `notebooks/data/chars/`. Per tant el CSV
  **no és sortida de v2**.
- Recompute de l'algorisme exacte de v2 sobre els 341 crops `_box` → **74 acceptats**, no 90.
  La causa exacta de la diferència (filtres diferents al segmentador germà, `N_CHARS_MAX=8` vs 10,
  o acumulació de diverses execucions al directori de chars) queda **pendent d'investigar**.

---

## Secció 2 — Anàlisi de causes arrel (amb evidència)

Totes les xifres provenen de l'agregació a nivell de grup de `models/eval_real_results.csv`
(90 grups). Distribució del delta `n_segmented − gt_length`: `−2`:3, `−1`:6, `0`:51, `+1`:22,
`+2`:6, `+3`:2. Mètriques globals: **char accuracy (ED) 62.6%** (ed total 234 / 626 chars GT),
**plate accuracy 16%** (14/90 perfectes).

### 2.1 Falsos positius — 30/90 grups (33%), `n_segmented > gt_length`

- **Distribució**: la majoria són `+1` (22 grups); 6 són `+2`; 2 són `+3` (el pitjor és
  `eu10_box3`: 10 segmentats vs 7 GT).
- **El filtre de confiança NO els elimina**: dels caràcters extra als 30 grups FP, només **13** són
  marcats `filtered_out=True`, i només **6/30 grups** recuperen `n_after_filter == gt_length`. La
  confiança mediana dels retinguts és **0.994** i la dels filtrats **0.447**: és a dir, **24/30
  grups FP conserven blobs extra d'alta confiança** que el CNN classifica amb seguretat. → **Un
  filtre de confiança no pot resoldre els FP; són formes "caràcter-like" reals.**
- **Posició dels FP**: sense biaix fort (filtrats a inici 3 / mig 4 / final 6).
- **Geometria real (inspecció de crops)**:
  - `eu10_box3.png`, `eu11_box6.png`, `test_065_box1.png` tenen tots la **banda blava de la UE** a
    l'esquerra. Però geometricament només **3–5/30** grups FP tenen el blob esquerre dins del 12% de
    l'amplada i estret (AR<0.35). → la banda UE **és un contribuïdor, no el dominant**.
  - `test_065_box1.png` té el bounding box **massa ample**: inclou una línia de **text del
    concessionari sota la placa** ("BS auto Brno…"). Font d'FP no relacionada amb la placa.
  - El recompte [5,**10**] deixa passar tots aquests FP perquè 8–10 blobs encara cauen dins el rang.

### 2.2 Caràcters perduts — 9/90 grups (10%), `n_segmented < gt_length`

> Nota: el skill esmenta "15%"; la xifra real al CSV actual és **10% (9/90)**.

- **No tots són "I perduda"; molts són crops de soroll**. Prediccions dels 9 grups: `BBI7B`,
  `TIWHFF`, `JHHIE`, `TH5IIJ`, **`WWLWLW`** (`test_067_box0`), `FK9EA`, `PD335H`, `PD722B`, `NU52EA`.
- `test_067_box0.png` inspeccionat visualment: **és la graella del cotxe amb el logo Škoda, sense
  cap matrícula** (ed=7, tot incorrecte). Les lamel·les horitzontals generen 6 blobs → fals
  positiu **del detector**, no caràcters perduts.
- `test_084_box2.png` (`PD722BF`, GT 7, segmentats 6, **ed=1**) sí que és una placa real que **perd 1
  caràcter a la vora** (la `F` final). Causa: retall sense marge i/o caràcter tallat pel bounding box.
- **AR de la 'I'**: el blob estret real més extrem mesurat té **AR=0.071**, per sobre del floor
  `0.05`. Només 3/678 blobs acceptats tenen AR<0.10. → el floor **no és massa estricte per als
  caràcters que sí es detecten**, però **els caràcters perduts no apareixen a aquesta estadística**
  (no es detecten), així que no es pot descartar que algun s'estigui perdent abans del filtre AR.
  Pendent: mesurar sobre casos `−1` etiquetats a mà.

### 2.3 Dobles contorns amb `RETR_EXTERNAL`

`RETR_EXTERNAL` ja elimina els contorns interns de 'O','0','D','B' (decisió correcta de l'ADR-002).
Els grups `+1`/`+2` observats **no** són dobles del mateix caràcter sinó blobs nous (banda UE, text
veí, traços partits). Un caràcter trencat per la binarització en dos blobs externs sí que en
generaria un d'extra; cap evidència directa al CSV ho confirma de forma aïllada → **pendent
d'investigar** crop a crop.

### 2.4 Deskew per regressió sobre centroides

- **Gairebé mai s'activa**: 226/341 crops (66%) tenen <3 candidats a la passada 1 → angle=0 forçat.
  Angle medià 0.00°; només 16 crops superen 5°; 7 queden bloquejats pel límit de 15°.
- El límit `|angle|>15°` rarament mossega (7 casos). Queda **pendent** determinar, crop a crop, si
  aquests 7 són plaques legítimament inclinades >15° (que ara no es roten) o regressions
  espúries sobre crops d'FP.
- **Risc del mínim de 3 candidats**: si la passada 1 detecta exactament 3 i un és FP, la recta
  s'ajusta a 3 punts amb un outlier → angle poc fiable. Amb soroll, una mala estimació pot
  **empitjorar** la passada 2.

### 2.5 Distorsió i manca de marge al retall (causa probable d'errors d'OCR)

- v2 retalla `thresh2[y:y+h, x:x+w]` (**sense marge**) i fa `cv2.resize(..., (28,28))`
  directament. Com que el bbox no és quadrat, **l'aspect ratio es deforma** i el caràcter omple la
  imatge **vora a vora**.
- En canvi, els caràcters sintètics d'entrenament es renderitzen centrats en un llenç **64×64 amb
  marge** (`notebooks/ocr/01_render_fonts.py`, `MARGIN=6`) i es redueixen a 28×28 al final
  (`notebooks/ocr/02_augment.py:60,243`), **preservant l'AR i deixant vora**.
- Inspecció dels chars guardats: `eu10_box3_char6.png`, `eu11_box6_char0.png`,
  `test_084_box2_char0.png` es veuen **estirats i sense marge**, visualment diferents dels chars
  d'entrenament. → **mismatch train/inferència imputable al segmentador**, probable contribuïdor
  d'errors d'OCR fins i tot quan la segmentació és correcta (els 51 grups exactes).

### 2.6 Manca de `is_plausible_plate`

- Avui l'únic filtre de qualitat global és el recompte [5,10]. No hi ha cap comprovació de
  coherència espacial.
- Recompute d'un `is_plausible_plate` proposat (std d'alçades ≤20% de la mediana, std de `cy` ≤25%,
  amplada ocupada ≥30% del crop) sobre els 74 crops que v2 accepta → **en rebutjaria 7**. Targeta
  precisament els crops de soroll coherent en recompte però incoherents en geometria (tipus
  `test_067_box0`). Impacte modest però dirigit; cal calibrar els llindars per no rebutjar plaques
  reals amb una `I` (alça la std d'alçades).

---

## Secció 3 — Anàlisi crítica de la tècnica

### 3.1 La tècnica de contorns és la correcta?

Sí per a aquest dataset: és robusta a la il·luminació no uniforme (adaptiveThreshold) i no necessita
`n_chars` a priori, a diferència de la projecció vertical. El **33% d'FP no és el sostre de la
tècnica sinó dels filtres**: el problema no és la detecció de contorns sinó que (a) el recompte
[5,10] és massa ample, (b) no hi ha filtre de coherència geomètrica, i (c) molts FP entren com a
crops espuris del detector. Aplicant totes les millores pendents (`is_plausible_plate`, `N_CHARS_MAX`
realista=8, padding/AR, dedup IoU) el sostre raonable estaria al voltant de **reduir els 30 grups FP
a la meitat** i recuperar els `−1` de vora (p. ex. `test_084`), però **no resol els crops d'FP del
detector** (`test_067_box0`), que són responsabilitat de la fase 1.

### 3.2 Alternatives reals (esforç / benefici concret / risc)

- **`connectedComponentsWithStats` en lloc de `findContours`** — Esforç ~3h. Benefici: maneig més
  net de blobs i estadístiques (àrea, centroide) gratuïtes per a `is_plausible_plate`. No recupera
  per si sol cap placa concreta. Risc baix. → marginal; només si ja es toca la detecció de blobs.
- **Contorns + projecció vertical sobre la placa ja alineada (passada 2)** — Esforç ~5h. Benefici:
  la projecció podria separar caràcters tocant-se i confirmar gaps; ajudaria en alguns `+1` per
  traços partits. Risc mig (dues fonts de veritat a reconciliar). → experimental.
- **Repassar el detector VJ / adoptar `main_3.py` i re-afinar** — Esforç 4–6h. Benefici concret:
  eliminar crops com `test_067_box0` (graella) **en origen** i ajustar bounding boxes massa amples
  (`test_065_box1`) reduiria FP i crops-soroll abans de segmentar. Risc mig/alt (afinar pot baixar
  recall). → probablement **més rendible que filtrar a posteriori**.
- **Passada 0 de quality-check del crop sencer** (contrast, mida de blobs, AR del crop) — Esforç
  ~3h. Benefici: descarta graelles/textures abans de binaritzar. Risc mig. → complementa §2.6.

### 3.3 Què està fora del segmentador

- Errors `1→7` (l'error més freqüent, documentat a `notebooks/ocr/CLAUDE.md`) i `8→B` són del
  **CNN** (domain gap del dataset sintètic), **no del segmentador**.
- Crops d'FP pur (graella `test_067_box0`) són del **detector VJ**.
- Plaques amb il·luminació molt dolenta poden no ser resolubles sense canviar la captura.

---

## Secció 4 — Pla d'acció prioritzat

Ordenat per impacte/esforç.

| # | Acció | Causa arrel | Impacte estimat | Esforç | Risc | Categoria |
|---|---|---|---|---|---|---|
| 1 | Connectar v2 al detector actual (arreglar glob `*_box*.png`→`_cand`/`.jpg`, rutes) i **regenerar el CSV** amb v2 | §1.5: el CSV no és de v2; v2 no llegeix `data/processed/` | Obté el **primer baseline real de v2** sobre 519 crops (avui desconegut); desbloqueja tota decisió posterior | 1–2 h | Baix (sense canvi d'algorisme); pot revelar números pitjors | QUICK_WIN |
| 2 | Corregir distorsió d'AR + afegir marge al retall de caràcter (quadrar + padding abans del resize 28×28) | §2.5: retall estirat vora-a-vora vs entrenament centrat amb marge | Redueix el domain gap en **tots** els chars ben segmentats (≥51 grups exactes) → menys errors d'OCR | 1–2 h | Baix; cal replicar exactament el preprocessat d'entrenament | QUICK_WIN |
| 3 | Afegir `is_plausible_plate` (std alçades ≤20%, std `cy` ≤25%, amplada ≥30%) + baixar `N_CHARS_MAX` a 8 | §2.1/§2.6: 24/30 FP són alta-confiança, el filtre de conf no els treu; recompte massa ample | Ataca els **30 grups FP**; recompute rebutja 7/74 crops de soroll | 2–3 h | Mig: pot rebutjar plaques reals amb `I`; calibrar llindars | STRUCTURAL |
| 4 | Reduir FP **en origen** al detector (afinar AREA/ASPECT/EXTENT o adoptar `main_3.py`) | §2.1/§2.2: crops massa amples (`test_065_box1`) i crops sense placa (`test_067_box0`) | Menys crops espuris i menys "plaques" de soroll abans de segmentar | 4–6 h | Mig/Alt: afinar pot baixar recall de plaques reals | STRUCTURAL |
| 5 | Padding del crop abans de binaritzar per no perdre caràcters de vora | §2.2: `test_084_box2` perd la `F` final (ed=1) | Recupera uns pocs grups `−1` de vora | 1 h | Baix | QUICK_WIN |
| 6 | Passada 0 de quality-check del crop sencer (contrast/AR/mida de blobs) | §2.2: graella `test_067_box0` segmentada com a placa | Descarta crops-textura tipus graella; complementa #3 | ~3 h | Mig: llindars massa durs descarten plaques fosques | EXPERIMENTAL |
| 7 | Reforçar l'estimació de deskew (RANSAC sobre centroides; revisar mínim=3 i límit 15°) | §2.4: 66% dels crops no s'intenten corregir; 3 punts amb 1 outlier → angle poc fiable | Baix retorn immediat (angle medià 0°, només 16 crops >5°) | 3–4 h | Baix payoff; risc baix | EXPERIMENTAL |
| 8 | (No tocar el segmentador) Domain gap del CNN: `1→7`, `8→B` | §3.3: error del classificador, no de la segmentació | Fora d'abast d'aquesta auditoria; tractar a `notebooks/ocr/` | — | — | OUT_OF_SCOPE |

---

## Secció 5 — Conclusió i recomanació

**1. El segmentador té sostre amb les millores pendents o cal reformar-lo?** No cal reformar-lo. La
tècnica de contorns de dues passades és adequada; el 33% d'FP prové dels **filtres massa laxos**
(recompte fins a 10, sense coherència geomètrica) i de **crops espuris del detector**, no de la
detecció de contorns en si. Amb les accions #2, #3 i #5 el segmentador hauria de millorar
substancialment sense canviar de paradigma.

**2. Quina acció té més retorn immediat amb menys risc?** L'**acció #1** (connectar v2 al detector
actual i regenerar el CSV) i l'**acció #2** (corregir la distorsió d'AR del retall). #1 és
imprescindible perquè **avui no tenim cap mètrica real de v2** — totes les xifres de §2 són d'un
segmentador germà. #2 és barata, de baix risc i beneficia tots els caràcters ben segmentats.

**3. Quina és la decisió més arriscada?** L'**acció #4** (afinar el detector VJ). És la de més
impacte potencial sobre els FP, però afinar els llindars de detecció pot reduir el recall de plaques
reals; cal fer-ho amb un conjunt de validació i mesurant abans/després.

**4. Hi ha inversió fora del segmentador amb més impacte total?** Sí, dues:
   - **Detector (fase 1)**: eliminar crops com la graella `test_067_box0` i els bounding boxes massa
     amples redueix FP i soroll *abans* de segmentar — sovint més rendible que filtrar a posteriori.
   - **CNN (OCR)**: l'error `1→7` és el més freqüent i és domain gap del dataset sintètic; reduir-lo
     (millor augmentation, fine-tuning amb GT real) pot pujar la char accuracy més que qualsevol
     ajust del segmentador, atès que el 57% dels grups ja se segmenten amb el recompte correcte.

> **Recomanació d'ordre d'execució:** #1 → #2 → #3 → #5, mesurant el CSV després de cada pas; només
> llavors decidir sobre #4 (detector) amb dades de v2 reals a la mà.

---

### Annex — punts marcats com a *pendent d'investigar*
- Causa exacta de la diferència recompute(74) vs CSV(90 grups) (§1.5).
- Si els 7 crops amb angle clamped >15° són plaques legítimes o FP (§2.4).
- Si el floor AR=0.05 perd alguna `I` real abans del filtre (cal etiquetar a mà els grups `−1`, §2.2).
- Confirmació crop-a-crop de dobles contorns externs per traços partits (§2.3).
