# ADR-005 — Format de sortida del caràcter 28×28 (`vj` vs `regla-or`)

- **Estat:** ✅ Adoptat (`vj`)
- **Fase:** 2 ↔ 3 — Interfície segmentador/OCR
- **Mètrica oficial:** char accuracy (ED) **67.5%** (`vj`) vs **37.7%** (`regla-or`)

---

## Context

La sortida del segmentador (ADR-002) és la **entrada** de l'OCR (ADR-004): una imatge 28×28
binària, caràcter blanc sobre fons negre. La manera exacta de construir aquesta imatge a partir
del bbox del caràcter determina si hi ha **train/test skew** — divergència silenciosa entre com
es van renderitzar les imatges d'entrenament i com el segmentador normalitza les de test. Aquest
és el risc més perillós del pipeline, perquè **no apareix a la matriu de confusió de train** però
destrossa l'accuracy real.

## Decisió

S'adopta el format **`vj`: retall binari del caràcter + `resize` directe a 28×28**, sense
recentrat ni reconstrucció del llenç.

## Alternatives considerades

Les dues es van implementar i mesurar amb la CNN real sobre el dataset.

### Alternativa A — `regla-or` (llenç 64×64 + marge)

Reconstrueix el caràcter en un llenç amb el mateix marge relatiu que el render de fonts
(`01_render_fonts.py`), replicant el procés d'entrenament *fil per randa*. La hipòtesi teòrica
era que calia replicar exactament la convenció del llenç d'entrenament.

### Alternativa B — `vj` (resize directe) ✅ **Adoptada**

Agafa el retall binari del caràcter i el redimensiona directament a 28×28.

| Format de sortida | Char accuracy (ED) | Plate accuracy | Char accuracy (posicional) |
|---|---|---|---|
| `regla-or` (llenç 64×64 + marge) | 37.7% | 10.1% | 58.1% |
| **`vj`** (resize directe) ✅ | **67.5%** | **27.9%** | **87.5%** |

## Justificació

La decisió és **empírica i contraintuïtiva**. La teoria deia que calia replicar el llenç de
l'entrenament (`regla-or`); la mesura va demostrar que el format simple de resize directe (`vj`)
funciona **gairebé el doble de bé** en char accuracy (67.5% vs 37.7%) i gairebé triplica la plate
accuracy (27.9% vs 10.1%).

Aquest és l'exemple paradigmàtic del principi de "valorar solucions amb dades, no amb
intuïcions": l'script d'avaluació del segmentador existeix precisament per mesurar variants com
aquesta de forma objectiva en lloc de decidir a priori. La regla pràctica que en queda: davant de
dues convencions de normalització, **es tria la que la mesura amb la CNN real validi**, no la que
sembli teòricament més correcta.

## Conseqüències

**Positives:**
- Char accuracy gairebé doblada respecte de l'alternativa teòricament "correcta".
- Format més simple d'implementar i mantenir.

**Negatives / riscos:**
- El format guanyador trenca la intuïció de "replicar exactament l'entrenament", cosa que cal
  documentar bé perquè un futur desenvolupador no el "corregeixi" tornant a `regla-or`.
- Persisteix un risc latent de skew si el pipeline d'entrenament canvia la convenció de render
  sense reavaluar aquest format.

## Referències

- `docs/situacio-actual.md` §4.1; ADR-002 (segmentació), ADR-004 (OCR)
