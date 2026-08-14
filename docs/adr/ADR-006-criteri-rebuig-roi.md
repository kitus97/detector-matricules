# ADR-006 — Criteri de rebuig de ROIs no-matrícula

- **Estat:** ✅ Adoptat
- **Fase:** 2 — Segmentació
- **Mètrica oficial:** **79 acceptades / 262 rebutjades** sobre 341 crops d'entrada

---

## Context

El detector (ADR-001) opera amb filosofia de màxim recall i, per disseny, retorna falsos
positius (banda blava de la UE, text del concessionari, textures regulars del cotxe). El
segmentador és l'**última oportunitat clàssica** de filtrar aquests falsos positius abans
de l'OCR: si una ROI no és realment una matrícula, s'ha de descartar i no generar caràcters.
Com que el dataset barreja formats de matrícula, no es pot validar per patró estricte
(NNNN-LLL); el criteri ha de ser geomètric i estructural.

## Decisió

Una ROI s'accepta només si supera **dos controls**:

**1. Recompte de caràcters dins del rang vàlid [5, 9].** Menys de 5 o més de 9 components
vàlids després del filtre geomètric → rebuig.

**2. Coherència geomètrica de fila** (`is_plausible_plate`), sobre els bboxes detectats:

| Comprovació | Llindar | Justificació |
|---|---|---|
| Desviació estàndard d'alçades | ≤ 20% de la mediana | Els caràcters reals tenen alçades molt uniformes |
| Desviació estàndard de `cy` (línia base) | ≤ 25% de la mediana | Els caràcters reals comparteixen línia base |
| Amplada total ocupada | ≥ 30% de l'amplada del crop | Una matrícula real ocupa la major part del crop |

Si la ROI no supera els controls, es descarta **sencera** sense guardar cap caràcter.

## Justificació

El recompte sol no n'hi ha prou: una textura de soroll pot generar casualment entre 5 i 9 blobs
de mida de caràcter. Els tres filtres de coherència afegeixen l'exigència que aquests blobs
formin una **fila coherent** (alçades uniformes, línia base compartida, cobertura horitzontal),
que és el que distingeix una matrícula real d'un fals positiu texturat.

El rang **[5, 9]** cobreix qualsevol format de matrícula del dataset sense ser tan ampli que
deixi passar soroll. Tots els llindars són **relatius** (a la mediana o a l'amplada del crop),
mai en píxels absoluts, per ser invariants a l'escala de la placa.

## Resultats (341 crops)

- 79 acceptades, 262 rebutjades, 582 caràcters generats.
- Motius de rebuig: `pocs_caracters` 225, `baseline_inconsistent` 26, `massa_caracters` 8,
  `alcades_inconsistents` 3.

## Conseqüències

**Positives:**
- Filtra la gran majoria dels falsos positius del detector sense aprenentatge ni patró fix.
- El motiu de rebuig queda registrat, cosa que permet auditar i ajustar els llindars amb dades.

**Negatives / riscos:**
- Falsos positius amb soroll **espacialment coherent** (textures molt regulars amb 5–9 blobs
  alineats) poden passar els controls. El guardià addicional previst és un **filtre de confiança
  post-OCR** basat en el `score` del Softmax: com que no hi ha format fix, no es pot validar per
  patró, només per confiança per caràcter. Documentat com a TODO (treball futur).

## Nota de coherència

El rang oficial és **[5, 9]**. Un esborrany inicial fixava [5, 8]; el criteri vigent és [5, 9],
d'acord amb `docs/situacio-actual.md`.

## Referències

- `docs/situacio-actual.md` §4; ADR-002 (segmentació), ADR-004 (filtre de confiança futur)
