# Registre de decisions d'arquitectura (ADR) — Sistema ALPR

> Recull de les decisions tècniques rellevants preses durant el desenvolupament del
> sistema de lectura automàtica de matrícules (ALPR) amb visió clàssica + una CNN d'OCR
> entrenada des de zero. Cada ADR documenta una decisió, les alternatives considerades, la
> justificació i les conseqüències. L'estat reflecteix la solució **oficial** validada
> empíricament a `docs/situacio-actual.md` (juny 2026).

## Format

Cada ADR segueix l'estructura: **Context · Decisió · Alternatives considerades · Justificació
· Conseqüències**. Quan una decisió tenia més d'una solució implementada, es comparen de
forma simètrica i s'indica explícitament quina ha quedat com a oficial i amb quina mètrica.

## Índex

| ADR | Títol | Fase | Estat |
|---|---|---|---|
| [ADR-001](ADR-001-detector-morfologic.md) | Detector de matrícules per morfologia (Sobel vertical) | 1 — Detecció | ✅ Adoptat |
| [ADR-002](ADR-002-segmentacio-contorns.md) | Segmentació de caràcters per contorns | 2 — Segmentació | ✅ Adoptat |
| [ADR-003](ADR-003-deskew-centroides.md) | Correcció d'inclinació per regressió de centroides | 2 — Segmentació | ✅ Adoptat |
| [ADR-004](ADR-004-ocr-cnn-fonts-sintetiques.md) | OCR amb CNN entrenada amb fonts sintètiques | 3 — OCR | ✅ Adoptat |
| [ADR-005](ADR-005-format-sortida-28x28.md) | Format de sortida del caràcter 28×28 (`vj` vs `regla-or`) | 2 ↔ 3 — Interfície | ✅ Adoptat |
| [ADR-006](ADR-006-criteri-rebuig-roi.md) | Criteri de rebuig de ROIs no-matrícula | 2 — Segmentació | ✅ Adoptat |
| [ADR-007](ADR-007-correccio-contextual-reader.md) | Lectura final sense correcció contextual (dataset sense patró) | 3b — Reader | ✅ Adoptat (revisat) |

## Nota sobre les discrepàncies de codi

Durant el desenvolupament van quedar al codi un parell de referències obsoletes que **NO**
reflecteixen el sistema final. La font de veritat és `docs/situacio-actual.md`:

- **Nombre de classes de l'OCR**: el sistema oficial té **36 classes (A–Z, 0–9)**. Alguns
  comentaris del codi parlen de "62 classes" (que inclourien minúscules) per herència de la
  fase de disseny. El model entrenat (`models/char_cnn_best.pth`) té 36 sortides.
- **Rang de caràcters vàlid per placa**: l'oficial és **[5, 9]**. Algun esborrany inicial
  fixava [5, 8]; el criteri vigent és [5, 9].
