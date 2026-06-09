"""
alpr/reader/reader.py
=====================
Fase 3b — Reader: munta el string final de la matrícula a partir dels caràcters
predits per l'OCR.

IMPORTANT — aquest dataset NO segueix cap patró. Les matrícules poden ser de
qualsevol país europeu (Sèrbia, Eslovènia, Turquia, Espanya…) i contenen lletres
(A–Z) i dígits (0–9) en MAJÚSCULES, de 5 a 9 caràcters, en ordre **totalment
aleatori**. Cap posició té un tipus esperat (lletra o dígit).

Conseqüència de disseny: NO es pot aplicar cap correcció contextual de confusions
(O↔0, S↔5, I↔1…), perquè no hi ha manera de saber en quina direcció corregir sense
un patró posicional — i corregir a cegues introduiria errors nous. El reader, per
tant, es limita a **netejar i concatenar** la millor predicció de l'OCR.
"""

from __future__ import annotations

import re


def read_plate(chars: list[str]) -> str:
    """
    Munta la matrícula final: concatena els caràcters, passa a majúscules i
    elimina '?' (baixa confiança) i qualsevol símbol no alfanumèric.

    No aplica cap correcció contextual: el dataset no té format fix (veure la
    capçalera del mòdul), així que la predicció de l'OCR es retorna tal qual,
    només netejada.
    """
    text = "".join(chars).upper()
    return re.sub(r"[^A-Z0-9]", "", text)
