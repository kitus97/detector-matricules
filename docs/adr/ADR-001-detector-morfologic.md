# ADR-001 — Detector de matrícules per morfologia (Sobel vertical)

- **Estat:** ✅ Adoptat
- **Fase:** 1 — Detecció
- **Mètrica oficial:** Recall (centre del GT dins d'algun candidat) **98.1%** (106/108)

---

## Context

La primera fase del pipeline ha de localitzar regions candidates (ROIs) que continguin una
matrícula dins de la foto completa del cotxe. Una restricció dura del projecte és que **no es
poden usar models de detecció preentrenats**: la detecció ha de ser implementació pròpia. Cal,
doncs, una tècnica que exploti una propietat estructural de la matrícula sense aprenentatge
sobre dades externes.

La propietat clau identificada: una matrícula és *una regió rectangular horitzontal amb una
densitat anormalment alta de vores verticals*, generada pels traços dels caràcters. El fons,
el para-xocs i els llums no presenten aquesta densitat de vores verticals concentrada.

## Decisió

S'adopta un **detector morfològic basat en gradient vertical**:

```
gris → suavitzat Gaussià → CLAHE → Sobel vertical → binarització Otsu
     → closing horitzontal + opening → components connexos → filtre de forma
```

El **closing horitzontal** fusiona les vores verticals properes dels caràcters en un únic
blob rectangular compacte; el filtre de forma (àrea, aspect ratio, extensió, mida) selecciona
els blobs amb geometria de placa.

> Nota d'implementació: una versió anterior d'aquest ADR esmentava una restricció addicional
> de la cerca a la franja vertical amb més activitat Sobel (projecció de files). Aquesta passa
> **no s'ha implementat** (ni al notebook `01_morphologic_VJ.ipynb` ni a `alpr/detector/`); el
> filtre de forma per components connexos és l'únic mecanisme de selecció. Els falsos positius
> els filtra la Fase 2 (ADR-006).

**Filosofia de disseny — màxim recall:** és preferible retornar diversos candidats (mediana 3,
mitjana 3.5 per imatge) i **garantir que la matrícula real hi sigui sempre**, deixant que les
fases posteriors (segmentador) filtrin els falsos positius. És millor tenir 3 candidats amb 1
de bo que 0 candidats.

## Alternatives considerades

| Alternativa | Per què es va descartar |
|---|---|
| **Harris / SIFT** (keypoints) | Útils com a descriptor de densitat de cantonades, però no aprofiten directament la propietat estructural "blob horitzontal dens en vores"; menys directes per localitzar. |
| **HOG + SVM** | Alternativa vàlida, però **supervisada** (cal centenars de positius + milers de negatius etiquetats), requereix finestra lliscant multi-escala (computacionalment cara) i és sensible a la rotació. Es va plantejar com a comparació per a la memòria, però la morfologia dona millors resultats i és més simple. |
| **Viola-Jones / Haar cascade** | Es va considerar com a verificador opcional de candidats; no es va adoptar com a detector principal. |

## Justificació

La morfologia + filtres de forma:

- Captura **directament** la propietat estructural de la matrícula (Sobel vertical + closing
  horitzontal).
- **No depèn del contingut** concret dels caràcters.
- És **robusta a perspectiva moderada** i a inclinacions lleugeres.
- És **computacionalment barata** (operacions lineals i morfològiques).
- **No requereix entrenament** ni dades etiquetades, cosa que respecta la restricció del
  projecte i evita el cost de muntar un dataset supervisat.

## Conseqüències

**Positives:**
- Recall del 98.1% (centre del GT) amb només ~3.5 candidats per imatge — l'objectiu de la
  fase (no perdre cap matrícula) s'assoleix.
- Pipeline 100% propi, sense dependència de detectors externs.

**Negatives / riscos:**
- El recall amb IoU ≥ 0.5 baixa al 79.6% (86/108): la caixa no sempre és perfectament
  ajustada. La Fase 2 ho corregeix amb el seu propi deskew i normalització.
- Els **falsos positius** (banda blava de la UE, text del concessionari, textures regulars
  del cotxe) són inherents a l'enfocament morfològic i no s'eliminen del tot en aquesta fase;
  és responsabilitat del criteri de rebuig del segmentador (ADR-006) filtrar-los.

## Referències

- `notebooks/01_*` (detector), `docs/situacio-actual.md` §3
- Material del curs: Sobel/gradient, morfologia matemàtica, Viola-Jones
