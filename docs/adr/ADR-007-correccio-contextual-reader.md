# ADR-007 — Lectura final sense correcció contextual (dataset sense patró)

- **Estat:** ✅ Adoptat · **revisat 2026-06-09** (abans descrivia una correcció contextual;
  veure "Revisió" al final)
- **Fase:** 3b — Reader

---

## Context

La sortida de l'OCR (ADR-004) presenta errors sistemàtics de **domain gap** entre parells de
caràcters visualment semblants: `0↔O`, `1↔I`, `5↔S`, `2↔Z`, `8↔B`. Inicialment es va plantejar
que el reader podria recuperar aquests errors aplicant una **correcció contextual posicional**
(corregir cap a lletra o cap a dígit segons la posició dins del patró de matrícula).

**Aquesta premissa, però, és falsa per a aquest dataset.** Les matrícules **no segueixen cap
format ni patró fix**: poden ser de qualsevol país europeu (Sèrbia, Eslovènia, Turquia,
Espanya…), contenen només **A–Z + 0–9 en majúscules**, tenen **de 5 a 9 caràcters** i estan en
**ordre totalment aleatori**. **Cap posició té un tipus esperat** (lletra o dígit).

## Decisió

El reader és un **muntador pur**: neteja i concatena la millor predicció de l'OCR, **sense cap
correcció contextual**.

```python
def read_plate(chars: list[str]) -> str:
    text = "".join(chars).upper()
    return re.sub(r"[^A-Z0-9]", "", text)   # elimina '?' i símbols no alfanumèrics
```

## Alternativa considerada i descartada

### Correcció contextual posicional — ❌ descartada

Era la idea original d'aquest ADR: en posicions de lletra corregir `0→O, 1→I, 5→S, 2→Z, 8→B`; en
posicions de dígit, a l'inrevés. **Es descarta** perquè requereix conèixer el **tipus esperat de
cada posició**, i en un dataset sense format fix això no existeix: davant d'una `S` no hi ha manera
de saber si hauria de ser `5` o no. Qualsevol correcció a cegues **introduiria errors nous** (p. ex.
forçar el format espanyol 4+3 convertia la lectura gairebé correcta `WAS666O` en `W456GGO`).

## Justificació

Principi **"no fer mal"**: sense un patró que indiqui el tipus de cada posició, la predicció de
l'OCR (amb la seva confiança) és la millor estimació disponible, i el reader no la pot millorar de
manera fiable. Limitar el reader al muntatge evita degradar lectures correctes.

## Conseqüències

**Positives:**
- El reader mai degrada una lectura correcta.
- Comportament simple, determinista i sense supòsits de domini incorrectes.

**Negatives / riscos:**
- Les confusions sistemàtiques de l'OCR (`O↔0`, `S↔5`…) **no es recuperen** a la lectura: queden
  com a errors de l'OCR. La via correcta per reduir-les és la **Fase 3** (millor augmentation o
  fine-tuning amb caràcters reals etiquetats), no el reader.

## Revisió (2026-06-09)

Versió anterior d'aquest ADR descrivia una correcció contextual posicional (lletra↔dígit segons el
patró). Es va detectar a l'auditoria de migració (`docs/auditoria-migracio.md`, troballa C4) que la
implementació forçava el format espanyol 4+3 i **corrompia** matrícules no espanyoles. En revisar-ho
amb l'usuari es va confirmar que **el dataset no té cap patró**, de manera que la correcció
contextual no és aplicable. L'ADR s'ha reescrit en conseqüència.

## Referències

- `docs/situacio-actual.md` §6; `docs/auditoria-migracio.md` (C4); ADR-004 (errors de l'OCR)
