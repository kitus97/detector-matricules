# Convencions de codi — projecte ALPR (detector-matrícules)

> Guia d'estil i convencions de programació del projecte. Pren com a base **PEP 8** i
> **PEP 257** (docstrings), i hi afegeix les normes específiques d'aquest repositori
> (idioma, OpenCV, organització). L'objectiu és que tot el codi `.py` tingui un aspecte
> homogeni, sigui llegible i mantenible, i respecti les decisions ja preses.
>
> Aquest document descriu *com* s'escriu el codi; per a *on va cada cosa* veure
> `docs/migracio-notebooks-a-py.md`.

---

## 1. Idioma (regla d'or del projecte)

- **Comentaris, docstrings i tota sortida a l'usuari (`print`, `logging`, missatges
  d'error) en CATALÀ.**
- **Identificadors (variables, funcions, classes, mòduls) en ANGLÈS.** Excepció tolerada:
  el codi de Fase 2 (`segmentador.py`) ja té identificadors en català (`segmenta_caixa`,
  `binaritza_deteccio`, `coherencia_fila`…). En codi NOU prioritza l'anglès; en *editar*
  fitxers existents, **mantén l'idioma que ja hi havia** per coherència interna.

```python
# ✅ Bé
def detect_boxes(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detecta regions candidates de matrícula amb filosofia de màxim recall."""
    log.info(f"Detectades {len(boxes)} caixes candidates")

# ❌ Malament (missatge en anglès)
log.info(f"Found {len(boxes)} candidate boxes")
```

---

## 2. Format general (PEP 8)

| Tema | Regla |
|---|---|
| **Indentació** | 4 espais. Mai tabuladors. |
| **Longitud de línia** | Màxim **99 caràcters** (els scripts actuals ja s'hi ajusten). Docstrings/comentaris: ~88. |
| **Línies en blanc** | 2 entre funcions/classes de nivell superior; 1 entre mètodes. |
| **Codificació** | UTF-8 (cal per als accents catalans). |
| **Cometes** | `"` dobles per defecte (consistent amb els scripts actuals). |
| **Final de fitxer** | Una línia en blanc al final, sense espais finals (*trailing whitespace*). |
| **Operadors** | Espais al voltant: `a = b + c`, `x <= y`. |

Format automàtic recomanat (opcional però desitjable): **`black`** (line-length 99) +
**`ruff`** per al linting. Si s'adopten, documentar-ho a `pyproject.toml`.

---

## 3. Nomenclatura

| Element | Estil | Exemple |
|---|---|---|
| Mòdul / paquet | `snake_case`, curt | `shape_filter.py`, `detector/` |
| Funció / variable | `snake_case` | `filter_by_shape`, `aspect_ratio` |
| Constant | `UPPER_SNAKE_CASE` | `AR_MIN`, `OUTPUT_SIZE`, `N_CHARS_MAX` |
| Classe | `PascalCase` | `CharCNN` |
| "Privat" (intern de mòdul) | prefix `_` | `_iou`, `_bbox_de_banda`, `_TRANSFORM` |
| Variable d'imatge | sufix que indica l'espai de color | `img_bgr`, `gray`, `thresh`, `roi_bgr` |

- Res de noms d'una lletra excepte índexs de bucle curts (`i`, `k`, `x`, `y`) i
  dimensions (`h`, `w`).
- Evita abreviatures opaques; `candidates` millor que `cnds`.

---

## 4. Docstrings (PEP 257 + estil del projecte)

L'estil establert és **docstring detallat en català que explica el _perquè_** de cada pas
de visió per computador, no només el _què_. Format tipus Google (seccions `Args:` /
`Returns:` / `Raises:`).

### 4.1 Docstring de mòdul

Capçalera amb el nom del fitxer subratllat, descripció, i (si és executable) una secció `Ús`:

```python
"""
shape_filter.py
===============
Filtre geomètric de la Fase 1 (detecció). Donada la sortida de l'etiquetatge de
components connexos, descarta els candidats que clarament no poden ser una
matrícula (àrea, aspect ratio, extensió, mida). Els llindars són deliberadament
laxos per no perdre la matrícula real (filosofia de màxim recall).
"""
```

### 4.2 Docstring de funció

```python
def filter_by_shape(
    num_labels: int,
    stats: np.ndarray,
    img_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Conserva només els candidats amb descriptors de forma plausibles.

    No decideix QUIN candidat és la matrícula; només elimina els absurds
    (carrosseria sencera, soroll minúscul) i deixa passar diversos blobs.

    Args:
        num_labels: nombre d'etiquetes de connectedComponents (inclou el fons).
        stats:      matriu de cv2.connectedComponentsWithStats.
        img_shape:  (H, W) de la imatge original, per calcular l'àrea relativa.

    Returns:
        Llista de bounding boxes (x, y, w, h) en píxels de la imatge original.
    """
```

**Normes:**
- Primera línia: imperatiu, frase curta acabada en punt (*"Detecta…", "Filtra…", "Retorna…"*).
- Funcions públiques (no `_`): docstring complet amb `Args`/`Returns` quan aporti claredat.
- Funcions trivials/privades: docstring d'una línia n'hi ha prou.
- Explica el **perquè** dels passos de CV (per què Otsu, per què `RETR_EXTERNAL`, per què
  marge…). Aquest és el segell del projecte.
- Documenta sempre **l'espai de color i el dtype** de les imatges d'entrada/sortida quan no
  sigui obvi (p. ex. "binari 0/255, blanc sobre negre").

---

## 5. Type hints

- **Obligatoris a totes les funcions públiques**; recomanats a les privades.
- Sintaxi moderna (Python 3.12): `list[...]`, `tuple[...]`, `dict[...]`, `X | None`.
- Posa `from __future__ import annotations` a dalt de cada mòdul (com fan els scripts
  actuals) per permetre anotacions sense cost en temps d'execució.
- Tipus habituals al projecte: `np.ndarray` (imatges), `Path` (rutes), `nn.Module` (model).

```python
from __future__ import annotations
from pathlib import Path
import numpy as np

def predict(char_img: np.ndarray, model, idx_to_class: dict[int, str]) -> tuple[str, float]:
    ...
```

---

## 6. Imports

Ordre PEP 8, separats per línia en blanc:

```python
from __future__ import annotations

# 1. Biblioteca estàndard
import argparse
import logging
from pathlib import Path

# 2. Tercers
import cv2
import numpy as np
import torch

# 3. Locals del projecte
import config
from alpr.common.io import load_image, save_image
```

- **Imports absoluts** des de l'arrel del paquet (`from alpr.detector import detect`).
- No `from module import *` (excepte, si de cas, `import config` accedint per
  `config.NOM`; evita `from config import *`).
- No imports dins de funcions, tret de dependències pesades/opcionals (p. ex. `matplotlib`
  només quan es visualitza — patró ja usat als scripts d'OCR).

---

## 7. Constants i configuració

- **Cap número màgic dins de la lògica.** Tot paràmetre ajustable viu a `config.py` o,
  com a molt, com a constant `UPPER_SNAKE_CASE` a la capçalera del mòdul (estil de
  `segmentador.py`, perquè iterar sigui ràpid).
- Agrupa les constants amb capçaleres de secció comentades i justifica breument cada valor.

```python
# ─── Filtres geomètrics relatius a l'alçada de la placa (mai px absoluts) ───
AR_MIN = 0.05   # aspect ratio mínim — NO pujar: mataria la 'I' i la '1'
AR_MAX = 1.10   # per sobre sol ser soroll o caràcters fusionats
```

---

## 8. Separadors de secció (estil del projecte)

Els scripts actuals organitzen els fitxers llargs amb dos nivells de separador. Mantén-los:

```python
# ══════════════════════════════════════════════════════════════════════════════
# Secció principal (p. ex. "Detecció de contorns")
# ══════════════════════════════════════════════════════════════════════════════

# ─── Subsecció o grup de constants ──────────────────────────────────────────
```

---

## 9. OpenCV i NumPy (convencions de domini)

- **Imatges en BGR `uint8`** per defecte (convenció d'OpenCV) a tot el sistema. Indica al
  nom de la variable l'espai de color (`img_bgr`, `gray`, `thresh`).
- Caràcters de sortida del segmentador: **28×28, binari {0, 255}, blanc sobre negre**.
- No mutis les imatges d'entrada in-place si la funció no ho promet: treballa sobre
  `.copy()` (com fa `draw_candidates`).
- Documenta sempre la **polaritat** de les binaritzacions (`THRESH_BINARY_INV` →
  blanc sobre negre) i els paràmetres clau (`blockSize`, `C`).
- Reproductibilitat: fixa les llavors a l'inici dels scripts amb aleatorietat
  (`random.seed`, `np.random.seed`, `torch.manual_seed`).

---

## 10. `print` vs `logging`

- **Codi de mòdul/biblioteca:** usa `logging`, no `print`.
- **Scripts d'entrada:** `logging` per al progrés; `print` només per a resums/taules
  finals formatats (patró ja usat a `04_evaluate_real.py`).
- Configuració estàndard del projecte:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

log.info(f"Crops processats: {n}")      # ✅ en català
```

---

## 11. CLI amb `argparse`

Tot script executable té un `main()` i un guard `if __name__ == "__main__":`. Arguments amb
**`help` en català**, valors per defecte des de `config`, i `choices` quan calgui:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Segmenta caràcters de matrícula.")
    parser.add_argument("--metode", choices=["contorns", "projeccio"], default="contorns",
                        help="Mètode de segmentació (default: contorns)")
    args = parser.parse_args()
    ...

if __name__ == "__main__":
    main()
```

- Rutes per defecte resoltes de manera robusta respecte al cwd
  (`Path(__file__).resolve().parents[N]`), com fa `segmentador.py`, perquè l'script
  funcioni des de qualsevol directori.

---

## 12. Funcions i disseny

- **Funcions pures** sempre que es pugui: reben dades, retornen dades, sense efectes
  secundaris ocults (escriure disc, mostrar finestres). L'I/O i la visualització es
  mantenen separats de la lògica de CV.
- Una funció hauria de cabre raonablement a la pantalla; si fa massa coses, parteix-la.
- Retorna tipus estables i documentats (no de vegades `None` i de vegades `list` sense
  documentar-ho). Per a "cap resultat", retorna llista buida `[]`, no `None`, quan
  semànticament sigui una col·lecció (p. ex. `segment()` retorna `[]` si rebutja la ROI).

---

## 13. Gestió d'errors

- Captura excepcions **específiques** (`ValueError`, `OSError`, `cv2.error`), mai `except:` pelat.
- A nivell de mòdul, deixa propagar; a nivell de script/bucle, captura i registra per no
  aturar tot el lot per una imatge dolenta:

```python
try:
    img = load_image(path)
except ValueError as e:
    log.warning(f"No s'ha pogut llegir {path.name}: {e}")
    continue
```

- Comprova sempre el retorn de `cv2.imread` (`None` ⇒ fitxer il·legible) abans d'usar-lo.

---

## 14. Higiene de notebooks

(Es mantenen com a exploració/registre; veure `notebooks/CLAUDE.md`.)

- Convenció de nom: `{nn}_{nom}.ipynb` on `nn` és l'ordre al pipeline.
- **Outputs nets abans de commitejar:** `jupyter nbconvert --clear-output --inplace *.ipynb`.
- La lògica reutilitzable no es queda al notebook: es porta al paquet `alpr/` i el notebook
  l'importa.

---

## 15. Entorn i dependències

- Python **3.12**, entorn `.venv` gestionat amb **`uv`**. Executa amb `uv run python …` o
  activant `.venv`.
- Si afegeixes una llibreria, posa-la a `requirements.txt` **i** `pyproject.toml` i
  **justifica-ho** al doc corresponent. (Recordatori: `torch`/`torchvision`/`Pillow` de
  l'OCR encara no hi són i caldrà afegir-los en oficialitzar el paquet.)
- Prioritza **OpenCV**; `scipy`/`scikit-image` permesos si aporten valor. Cap model
  preentrenat a detecció/segmentació.

---

## 16. Verificació

- No hi ha suite de tests ni CI: la **verificació és a ull** sobre les imatges de sortida i
  amb les mètriques (`models/eval_real_results.csv`, matriu de confusió, recall de detecció).
- Tota fase ha de poder generar **visuals de depuració** (caixes acceptades/rebutjades amb
  motiu) per facilitar la inspecció, com fa `segmentador.py` a `notebooks/data/debug/`.
- Si en el futur s'afegeixen tests, que siguin de funcions pures (filtres geomètrics,
  `iou`, `edit_distance`, `read_plate`), que són deterministes i fàcils d'aïllar.

---

### Resum de la filosofia

> Codi pla i llegible (PEP 8), **català de cara enfora i docstrings que expliquen el
> perquè**, anglès als identificadors nous, números a `config`, imatges BGR uint8 amb el
> color al nom, I/O i visualització separats de la lògica, i un `main()`+`argparse` net a
> cada script executable.
