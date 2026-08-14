# ADR-003 — Correcció d'inclinació per regressió de centroides

- **Estat:** ✅ Adoptat
- **Fase:** 2 — Segmentació (pas previ d'alineació)

---

## Context

El detector morfològic (ADR-001) no garanteix que la placa quedi perfectament horitzontal: el
recall amb IoU ≥ 0.5 és del 79.6%, cosa que indica caixes lleugerament desajustades o
inclinades. Una inclinació de 5–10° degrada els filtres geomètrics del segmentador (l'aspect
ratio dels caràcters deixa de ser real) i la qualitat del retall final 28×28. Cal corregir la
inclinació (deskew) **abans** de la segmentació estricta.

## Decisió

S'adopta un **esquema de dues passades amb regressió lineal sobre els centroides** dels
caràcters.

**Passada 1 (tolerant):** binarització adaptativa + detecció de contorns amb filtres laxos
(alçada 70–130% de la mediana, aspect ratio 0.05–1.2). L'objectiu no és segmentar bé, sinó
localitzar prou caràcters candidats per estimar l'angle.

**Estimació de l'angle:** es calcula el centroide `(cx, cy)` de cada bbox i s'ajusta una recta
per regressió:

$$m, b = \mathrm{polyfit}(cx_i,\ cy_i,\ 1) \qquad \alpha = \arctan(m)\cdot\frac{180}{\pi}$$

Si `|α| > 15°` o hi ha menys de 3 candidats, **no es rota** (senyal insuficient o detecció
errònia). La rotació s'aplica amb `warpAffine` al voltant del centre, amb `INTER_CUBIC` i
`BORDER_REPLICATE` per evitar franges negres als cantons.

**Passada 2 (estricta):** sobre la placa ja alineada, es repeteix la detecció amb els filtres
estrictes de l'ADR-002. Ara l'aspect ratio és geomètricament real.

## Alternativa considerada

### `minAreaRect` global — ❌ descartada

Aplicar `cv2.minAreaRect` sobre tots els píxels de primer pla del crop. Es va implementar
inicialment i es va descartar: el núvol global de punts (fons binaritzat, vores de la placa,
soroll) forma una forma aproximadament rectangular **alineada amb els eixos de la imatge**, de
manera que l'angle resultant és gairebé sempre **≈ 0°** independentment de la inclinació real.

## Justificació

Els caràcters d'una matrícula estan **alineats horitzontalment per disseny**, de manera que la
línia de centroides revela directament la inclinació de la placa — un senyal molt més net que
la forma global del crop. Per estimar un pendent no calen tots els caràcters perfectes: amb 4–5
centroides ben repartits la recta és fiable. La separació en dues passades aprofita aquesta
asimetria: la passada 1 només necessita prou candidats per estimar l'angle (toleranta), mentre
que la passada 2 exigeix condicions ideals per a la segmentació final (estricta).

## Conseqüències

**Positives:**
- Robust perquè es basa en el senyal rellevant (alineació dels caràcters), no en la geometria
  global del retall.
- Corregeix les inclinacions lleugeres que el detector deixa sense ajustar.

**Negatives / riscos:**
- L'esquema de dues passades aplica la binarització dues vegades per crop, incrementant el
  temps de processament. Acceptable per a un dataset acadèmic; caldria optimitzar per a
  producció.

## Alternativa no explorada

- **Transformada de Hough** per estimar l'angle: més robusta davant outliers, però més complexa
  i innecessàriament pesada per a inclinacions lleugeres (< 15°).

## Referències

- `docs/situacio-actual.md` §4; ADR-002 (segmentació)
