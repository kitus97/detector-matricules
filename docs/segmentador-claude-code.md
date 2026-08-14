# Segmentador de caràcters (Fase 02) — doc de treball

> Document viu del mòdul de segmentació construït segons `notebooks/fase_02/CLAUDE.md`.
> Conté: (1) el resum del context llegit, (2) les decisions de disseny i la seva
> justificació, (3) el pipeline final, (4) els paràmetres i el criteri, (5) limitacions,
> (6) com executar-ho, i (7) desviacions respecte al CLAUDE.md del mòdul.
>
> **Estat actual:** implementat i provat sobre els 341 crops de
> `notebooks/data/processed/`. Script: [`notebooks/fase_02/segmentador.py`](../notebooks/fase_02/segmentador.py).

---

## 1. Resum del context llegit (Pas 0)

### 1.1 `.claude/skills/segmentation-debug.md` (skill de depuració)

Descriu el pipeline de referència i 6 problemes coneguts amb les seves solucions:

- **Pipeline de dues passades**: passada 1 *tolerant* (detecta candidats per estimar
  l'angle de deskew per regressió sobre centroides) → passada 2 *estricta* (segmentació
  final + validació `is_plausible_plate`).
- **P1 — Dobles contorns** (O, 0, D, B…): `RETR_TREE` retorna contorn extern i forat
  intern → solució: `RETR_EXTERNAL` + xarxa de seguretat `remove_overlapping_bboxes` (IoU 0.5).
- **P2 — La 'I' no es detecta**: aspect ratio massa estret / traç fragmentat → solució:
  AR mínim **0.05** (no pujar-lo) + `MORPH_CLOSE` vertical `(1,3)` post-binarització.
- **P3 — Deskew sempre 0°**: `minAreaRect` global donava ≈0° → solució: regressió sobre
  centroides dels bboxes candidats.
- **P4 — Falsos positius del detector VJ**: 33% de grups amb `n_seg > gt_len`; el filtre
  de confiança de la CNN **NO** els elimina (conf. incorrectes 0.819 ≈ correctes 0.961).
- **P5 — Binarització amb il·luminació no uniforme**: Otsu global falla → `adaptiveThreshold`
  (MEAN_C, BINARY_INV, block=31, C=15); opcions: pujar block a 51, GAUSSIAN_C, CLAHE previ.
- **P6 — Caràcters retallats**: bbox massa just → `padding=2` al retall.
- Paràmetres de referència: `N_CHARS_MIN=5`, `N_CHARS_MAX=8`, `ALIGN_ANGLE_MAX=15°`,
  block=31, C=15, kernel CLOSE (1,3), sortida 28×28 **blanc sobre negre**.

### 1.2 `docs/adr/ADR-002-character-segmentation.md` (decisió d'arquitectura)

- **Decisió 1 — deskew per regressió sobre centroides** (no `minAreaRect` global).
  `warpAffine` amb `INTER_CUBIC` + `BORDER_REPLICATE`. No rota si <3 candidats o |α|>15°.
- **Decisió 2 — tècnica de segmentació**: **adoptada la detecció de contorns**
  (`adaptiveThreshold` MEAN_C/BINARY_INV/31/15 → `findContours` RETR_EXTERNAL,
  CHAIN_APPROX_SIMPLE → filtre per alçada mediana i AR). Es descarta la projecció vertical
  com a primària perquè requereix `n_chars` a priori i és sensible a il·luminació (Otsu global).
  La projecció queda com a **alternativa** al repositori.
- **Per què RETR_EXTERNAL**: elimina d'arrel els dobles contorns dels caràcters amb forat.
- **Filtre estricte**: alçada 85–115% de la mediana; AR **0.05–0.95** (el 0.05 és necessari
  per la 'I', AR real 0.05–0.12). `MORPH_CLOSE` (1×3) per consolidar la 'I'.
- **Decisió 3 — validació de qualitat**: recompte a **[5, 8]** + `is_plausible_plate`
  (std alçades ≤20% mediana, std `cy` ≤25% mediana, amplada total ≥30% del crop).
- Limitacions reconegudes: soroll que generi contorns mida-caràcter; la 'I' segueix sent
  el cas difícil.

### 1.3 `docs/problemes_segmentation.txt` (problemes detectats)

Quatre punts, ja coberts per l'skill/ADR: (1) dobles contorns 0/O/D; (2) no descarta totes
les no-matrícules (es preveu filtre OCR posterior); (3) la 'I' falla per AR (proposen CLOSE
als dígits); (4) dins d'una placa real de vegades cola soroll → es proposa filtre post-OCR.

### 1.4 `docs/segmenter_audit.md` (auditoria crítica) — **la més important per a decisions**

- **§1.5 (crític)**: el CSV `eval_real_results.csv` **NO** es va generar amb cap notebook
  de `fase_02`; prové d'un segmentador germà executat amb `cwd=notebooks/`. Les mètriques
  són la millor aproximació disponible, no una avaluació real de cap notebook concret.
- **§2.1**: 33% de grups amb falsos positius; el filtre de confiança no els treu (24/30
  FP conserven blobs de conf. mediana 0.994). La banda blava de la UE i text del
  concessionari sota la placa són fonts d'FP. El recompte `[5,10]` de v2 és massa ample.
- **§2.5 (CRÍTIC per a la sortida 28×28)**: v2 retallava `thresh[y:y+h, x:x+w]` **sense
  marge** i feia `cv2.resize(...,(28,28))` directe → **deforma l'aspect ratio** i omple
  vora a vora. Els sintètics d'entrenament es renderitzen **centrats en 64×64 amb
  MARGIN=6** i es redueixen a 28×28 al final, **preservant AR i deixant marge**. Això és
  un **train/test skew** imputable al segmentador → cal **quadrar + marge abans del resize**.
- **§2.6**: manca `is_plausible_plate`; l'únic filtre era el recompte.
- **Pla d'acció prioritzat**: #1 connectar al detector; **#2 corregir distorsió d'AR + marge
  (QUICK_WIN, baix risc, beneficia tots els chars ben segmentats)**; #3 `is_plausible_plate`
  + `N_CHARS_MAX`=8; #5 padding del crop per no perdre caràcters de vora.
- Fora d'abast del segmentador: errors `1→7`, `8→B` (domain gap de la CNN); crops d'FP pur
  (graella `test_067_box0`) són del detector.

### 1.5 `01_render_fonts.py` — **convenció de marge/escala/centrat (font de veritat)**

Com es generen les imatges base amb què s'entrena la CNN:

- Llenç **64×64**, escala de grisos, **fons BLANC (255), caràcter NEGRE (0)**.
- `MARGIN = 6` → el caràcter ocupa **com a màxim `64 − 2·6 = 52 px` en la dimensió més gran**.
- `fit_font` tria la mida de font perquè `max(char_w, char_h) ≤ 52`, **preservant l'aspect
  ratio natural** del glif (una '1'/'I' queda estreta; una 'W' queda ampla).
- `render_char` **centra el glif** al llenç 64×64 fent servir el bounding box del glif
  (`getbbox`): `x=(64−char_w)//2 − bbox[0]`, `y=(64−char_h)//2 − bbox[1]`. La dimensió més
  petita queda **centrada amb més marge**. → **És centrat geomètric per bounding box, NO
  centrat per centre de massa (MNIST).**
- La sortida d'aquest pas és **grisos anti-aliased, negre sobre blanc, SENSE binaritzar**.
  El docstring de `render_char` ho diu explícitament: la inversió i binarització es deixen
  per a l'augmentation, *"on s'aplicarà el mateix adaptiveThreshold que el segmentador real"*.
  → **El disseny espera que el segmentador real alimenti grisos (negre sobre blanc) pel
  mateix adaptiveThreshold.**

### 1.6 `02_augment.py` — **binarització exacta del "Grup D" (font de veritat)**

L'augmentation opera tot a **64×64** i aplica 4 grups en ordre estricte
`A (geometria) → B (traç) → C (contorn) → D (binarització)`; el **resize a 28×28 és l'últim pas**.

- **Grup D — `apply_binarize`** (sempre s'aplica), exactament:
  ```python
  cv2.adaptiveThreshold(
      img, 255,
      cv2.ADAPTIVE_THRESH_MEAN_C,
      cv2.THRESH_BINARY_INV,
      blockSize=31,
      C=15,
  )
  ```
  Com que el caràcter és negre (per sota del llindar) sobre blanc, `THRESH_BINARY_INV`
  el deixa **BLANC sobre fons NEGRE**.
- **Resize final**: `cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)`.
- **CONCLUSIÓ OPERATIVA (regla d'or):** la binarització es fa **a 64×64** (block=31 és
  relatiu a 64) i el **resize a 28 és l'ÚLTIM pas amb `INTER_NEAREST`**. **No s'ha de
  binaritzar directament a 28×28.** Replicar: glif grisos negre-sobre-blanc → centrat a
  64×64 amb marge (≈52px dim. gran) → `adaptiveThreshold(MEAN_C, BINARY_INV, 31, 15)` →
  `resize(28,28, INTER_NEAREST)`.

### 1.7 `03_train_cnn.py` i `04_evaluate_real.py` — model i `idx_to_class`

- **`CharCNN(n_classes)`** (a `03_train_cnn.py`): `Conv(1→32)+BN+ReLU+MaxPool` →
  `Conv(32→64)+BN+ReLU+MaxPool` → `Dropout(0.4)` → `Flatten` → `Linear(3136→256)+ReLU` →
  `Dropout(0.3)` → `Linear(256→n_classes)` (logits crus). Entrada **1×28×28**.
- El checkpoint `models/char_cnn_best.pth` guarda `model_state`, `n_classes`,
  `class_to_idx`, `val_acc`. `idx_to_class = {v:k for k,v in class_to_idx.items()}`.
- **Càrrega** (com `load_model` de `04_evaluate_real.py`): importar `CharCNN` via
  `importlib` des de `03_train_cnn.py` (el mòdul comença per dígit), reconstruir, carregar
  `model_state`, `eval()`. Device: cuda → mps → cpu.
- **`predict_char(model, img_path, device) -> (idx, conf)`** ja existeix amb la signatura
  demanada. El seu `_TRANSFORM` fa `Grayscale → Resize((28,28)) → ToTensor → Normalize(0.5,0.5)`.
  Com que guardarem PNG **exactament 28×28**, el `Resize` és un no-op → cap distorsió afegida.
  **Reutilitzaré aquesta funció i aquest transform tal qual (no els reinvento).**

---

## 2. Decisions de disseny (i justificació)

> *(Es completarà i ajustarà durant la implementació; aquí queden fixades les decisions de
> partida i el seu motiu.)*

| # | Decisió | Justificació |
|---|---|---|
| D1 | Mètode primari: `findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` sobre binari adaptatiu | ADR-002 l'adopta; RETR_EXTERNAL elimina dobles contorns |
| D2 | Projecció vertical com a alternativa activable per flag `--metode projeccio` | El CLAUDE.md i l'usuari volen poder **comparar** solucions |
| D3 | **Sortida 28×28: replicar render+GrupD, MAI MNIST** — glif grisos negre/blanc → centrat 64×64 (dim. gran ≈52px, AR preservat) → `adaptiveThreshold(MEAN_C,INV,31,15)` → `resize(28,INTER_NEAREST)` | REGLA D'OR (§1.5/§1.6); corregeix el train/test skew de §2.5 de l'auditoria |
| D4 | Llindars geomètrics **relatius a l'alçada de la placa** (mai px absoluts) | Exigència explícita del CLAUDE.md i de l'usuari |
| D5 | AR mínim **0.05** (no pujar) + `MORPH_CLOSE (1,3)` | Preservar la 'I' (P2/ADR) |
| D6 | Criteri de rebuig: **<5 o >9** components vàlids **O** fila incoherent (std alçades/`cy`, amplada total) | Secció 5 del CLAUDE.md del mòdul |
| D7 | Deskew per regressió sobre centroides (dues passades), reutilitzant la lògica de l'ADR | Decisió 1 de l'ADR; no `minAreaRect` global |
| D8 | Mode OCR opcional (`--ocr`) que reutilitza `predict_char` + `load_model` per importlib | Pas 3; no reinventar la càrrega del model |
| D9 | Normalitzar el crop a `NORM_H=96` abans de detectar (veure §2.1) | El `blockSize=31` fix falla amb plaques de 18–46 px; afegit durant la implementació |

**Desviació prevista respecte a l'ADR/skill:** el rang de recompte serà **[5, 9]** (no
[5, 8]) perquè és el que demana explícitament l'usuari a la tasca. Es documenta a §7.

---

### 2.1 Decisió afegida durant la implementació — normalització d'escala (D9)

**Problema detectat al primer smoke-test:** amb els paràmetres originals només
s'acceptaven 2/25 crops i es fragmentaven les plaques (eu1_box0: 42 blobs). **Causa
arrel:** les plaques reals d'aquest dataset són **molt petites** (mesures GT: eu10=18 px
d'alçada, eu11=28 px, eu1=46 px) i el `blockSize=31` de la binarització adaptativa és
**més gran que el crop sencer** → binaritza malament i sobre-segmenta.

**Decisió D9:** **normalitzar tots els crops a una alçada de treball comuna `NORM_H=96 px`**
(preservant l'aspect ratio) **abans de detectar**. Així el `blockSize=31` té sempre una
escala coherent respecte als caràcters. Com a efecte secundari positiu, upscalar
caràcters minúsculs millora la qualitat del 28×28. La sortida 28×28 s'extreu del gris
normalitzat i deskew-at (sense CLAHE), no del binari de detecció. **Resultat: 2→6
acceptades en el mateix subconjunt de 25 crops; eu1_box6 → `M5XSX` perfecte.**

## 3. Pipeline final

Per a cada `{stem}_box{n}.png`:

1. **Llegir** el PNG (BGR) → escala de grisos.
2. **Normalitzar l'escala** a `NORM_H=96 px` d'alçada, preservant l'AR (D9).
3. **Passada 1 (deskew):** CLAHE → `adaptiveThreshold(MEAN_C, INV, 31, 15)` →
   `MORPH_CLOSE(1,3)` → `findContours(RETR_EXTERNAL)` → filtre geomètric **tolerant** →
   `estima_angle` (regressió lineal sobre centroides) → angle de correcció.
4. **Rotar** el gris normalitzat amb `warpAffine(INTER_CUBIC, BORDER_REPLICATE)`.
5. **Passada 2 (segmentació):** re-binaritzar el gris rotat i detectar amb el **mètode
   triat** (contorns per defecte; projecció vertical amb `--metode projeccio`).
6. **Filtre geomètric estricte** relatiu a l'alçada de la placa (alçada, AR, àrea).
7. **Selecció de la fila dominant** per proximitat a la mediana d'alçades + **dedup IoU**
   + ordenació esquerra→dreta.
8. **Criteri de rebuig:** descarta la caixa si queden **<5 o >9** caràcters, o si la fila
   és **incoherent** (std d'alçades, std de `cy`, o amplada ocupada insuficients). Es
   registra el motiu.
9. **Sortida 28×28 (REGLA D'OR):** per a cada caràcter acceptat, glif en grisos
   (negre/blanc) → centrat en llenç 64×64 amb la dim. gran a 52 px (AR preservat) →
   `adaptiveThreshold(MEAN_C, INV, 31, 15)` → `resize(28, INTER_NEAREST)`. Desa a
   `notebooks/data/chars/{stem}_box{n}_char{k}.png`.
10. **Visual de depuració** a `notebooks/data/debug/{stem}_box{n}.png` (verd=acceptat,
    vermell=descartat, rètol amb angle/recompte/motiu).

## 4. Paràmetres triats i criteri

| Paràmetre | Valor | Criteri |
|---|---|---|
| `NORM_H` | 96 px | Escala de treball perquè `blockSize=31` sigui coherent amb els caràcters (D9, empíric) |
| `ADAPT_BLOCK` / `ADAPT_C` | 31 / 15 | Idèntics a l'ADR-002 i al Grup D (obligatori a la sortida) |
| `MORPH_CLOSE_KERNEL` | (1, 3) | Consolida el traç vertical de la 'I' (ADR/skill P2) |
| `DESKEW_MIN_CANDIDATS` / `_ANGLE_MAX` | 3 / 15° | ADR-002 (senyal insuficient / detecció errònia) |
| `H_CHAR_MIN_REL` / `_MAX_REL` | 0.30 / 1.05 | Filtre groller relatiu a H; deixa marge per a crops poc ajustats |
| `AR_MIN` / `AR_MAX` | 0.05 / 1.10 | 0.05 preserva 'I'/'1' (ADR P2, *no pujar*); 1.10 talla fusions/soroll |
| `AREA_MIN_REL` | 0.015 | Elimina speckle, relatiu a H² |
| `H_MED_TOL` | 0.30 | Conserva blobs amb alçada ∈ [0.70,1.30]·mediana (fila dominant) |
| `ROW_STD_H_MAX` / `_CY_MAX` | 0.22 / 0.22 | Coherència de fila (≈ valors de l'ADR `is_plausible_plate`) |
| `WIDTH_OCC_MIN` | 0.30 | La matrícula ocupa la major part del crop (ADR) |
| `N_CHARS_MIN` / `_MAX` | 5 / 9 | Petició de l'usuari (veure §7) |
| `CANVAS_SIZE` / `MARGIN` / `OUTPUT_SIZE` | 64 / 6 / 28 | **Idèntics a `01_render_fonts.py`** (regla d'or) |

> Els llindars geomètrics s'han fixat coherentment amb l'ADR-002 i validat sobre els 341
> crops; són ajustables a la capçalera de l'script per iterar. Cap és absolut en px.

## 5. Resultats i limitacions conegudes

### Resultats (341 crops de `notebooks/data/processed/`)

- **Mètode contorns (primari): 79 caixes acceptades**, 582 caràcters generats. Motius de
  rebuig: `pocs_caracters` 225, `baseline_inconsistent` 26, `massa_caracters` 8,
  `alcades_inconsistents` 3. (En línia amb el recompute de v2 de l'auditoria: 74.)
- **Mètode projecció (alternatiu): 40 acceptades** → **confirma l'ADR-002**: la detecció
  de contorns és netament superior a la projecció vertical en aquest dataset.
- **Validació de la regla d'or:** `eu1_box6` se segmenta net (M, 5, X, S, X; el guió "-"
  es rebutja correctament) i l'OCR llegeix **`M5XSX` amb confiança 1.00** (GT=`M5XSX`).
  Això demostra que el 28×28 coincideix amb la distribució d'entrenament quan la
  segmentació és correcta.
- **Avaluació global** (`04_evaluate_real.py` sobre els 79 grups): char accuracy (ED)
  37.7%, plate accuracy 10.1% (8/79 perfectes), char accuracy posicional 58% (36 grups
  alineats). **Aquesta xifra NO és comparable amb el "baseline" 62.6% de l'auditoria**:
  (a) l'auditoria §1.5 adverteix que aquell CSV no prové de cap segmentador de `fase_02`;
  (b) ara s'avaluen 41 grups que la regla antiga de longitud hauria descartat; (c) les
  caixes FP i la banda blava de la UE inflen la distància d'edició.

### Limitacions conegudes

- **Falsos positius del detector (fase 1).** Moltes caixes accepten blobs extra coherents
  (banda UE a l'esquerra, text del concessionari). L'auditoria §2.1/§3.3 conclou que el
  filtre de confiança de la CNN **no** els elimina i que són **responsabilitat del
  detector**, no del segmentador.
- **Biaix de gruix de traç (inherent, no és un bug).** Els caràcters generats tenen una
  fracció de blanc mediana de **0.28 vs 0.20** dels sintètics d'entrenament (≈ +40% de
  tinta). **Causa:** les plaques reals són minúscules (18–46 px) i upscalar-les a 52 px
  introdueix un halo de vora que `adaptiveThreshold` engreixa. Les fonts d'entrenament són
  nítides a 52 px; aquesta informació no existeix als crops petits, i extreure de
  resolució original no ho corregeix (l'original també és petit). Contribueix al *domain
  gap* sintètic→real (ja conegut: errors `1→7`, `8→B`). Es manté el Grup D **exacte** com
  mana el CLAUDE.md; mitigar-ho exigiria entrada de més resolució o re-entrenar l'OCR amb
  augmentation més gruixut (fora d'abast d'aquest mòdul).
- **Plaques molt petites** (p. ex. eu10, 18 px) se segmenten amb soroll tot i la
  normalització: poca informació de partida.

## 6. Com executar

Des de l'arrel del projecte (les rutes per defecte es resolen relatives a l'script, així
que funciona des de qualsevol cwd):

```bash
# Mètode primari (contorns). Neteja chars/ i debug/ i genera visuals + chars 28×28:
.venv/bin/python notebooks/fase_02/segmentador.py

# Mètode alternatiu (projecció vertical) per comparar:
.venv/bin/python notebooks/fase_02/segmentador.py --metode projeccio

# Llegir el resultat amb la CNN d'OCR (carrega models/char_cnn_best.pth):
.venv/bin/python notebooks/fase_02/segmentador.py --ocr

# Iteració ràpida sense esborrar res:
.venv/bin/python notebooks/fase_02/segmentador.py --limit 20 --no-net
```

**Flags:** `--metode {contorns,projeccio}`, `--ocr`, `--no-net` (no esborrar),
`--no-debug` (no generar visuals), `--limit N`. Rutes overridables:
`--processed-dir`, `--chars-dir`, `--debug-dir`.

**Interpretar els visuals** (`notebooks/data/debug/`): rètol superior amb
`{nom} ang=±X n=K ESTAT`. Caixes **verdes** = caràcters acceptats (numerats E→D); caixes
**vermelles** = blobs detectats però descartats; caixes **taronja** = caràcters d'una
caixa rebutjada. El motiu de rebuig apareix al rètol (`pocs_caracters`, `massa_caracters`,
`baseline_inconsistent`, `alcades_inconsistents`, `amplada_insuficient`).

**Per defecte es neteja `chars/` i `debug/`** a cada execució (decisió de l'usuari) per
evitar barrejar resultats de diferents segmentadors (avís de l'auditoria §1.5).

## 7. Desviacions respecte a `notebooks/fase_02/CLAUDE.md`

- **Recompte vàlid `[5, 9]`** en lloc de `[5, 8]` (petició explícita de l'usuari).
- **Normalització d'escala a `NORM_H=96`** abans de detectar (D9): no contemplada al
  CLAUDE.md, però necessària perquè el `blockSize` fix funcioni amb plaques molt petites.
  No afecta la regla d'or de la sortida (el 28×28 segueix replicant render+Grup D exactes).
- La binarització de **detecció** prova el camí CLAHE→adaptive (el CLAUDE.md permet provar
  Otsu/adaptive per a la detecció); la binarització de **sortida** és el Grup D exacte,
  sense desviació.
</content>
