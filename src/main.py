"""Demo Etapa A — generació de candidats a matrícula.

Ús:
    python main.py                       # processa unes imatges per defecte
    python main.py path/a/imatge.jpg     # processa una sola imatge
    python main.py path/a/directori/     # processa tot el directori
"""

import sys
from pathlib import Path

from pipeline.detector import generate_plate_candidates

# Imatges de mostra del dataset (relatives al directori de treball del repo).

DATA_DIR = Path(__file__).parent / "../data"

DEFAULT_SAMPLES = [
    DATA_DIR / "eu1.jpg",
    # DATA_DIR / "eu2.jpg",
    # DATA_DIR / "test_001.jpg",
    # DATA_DIR / "test_002.jpg",
    # DATA_DIR / "test_003.jpg",
    # DATA_DIR / "test_004.jpg",
    # DATA_DIR / "test_005.jpg",
    # DATA_DIR / "test_006.jpg",
    # DATA_DIR / "test_007.jpg",
]

DEFAULT_SAMPLES = list(DATA_DIR.glob("*.jpg"))

def main() -> None:
    if len(sys.argv) > 1:
        target = sys.argv[1]
        # results = generate_plate_candidates(target, verbose=True)
        results = generate_plate_candidates(target, verbose=True)
    else:
        results = {}
        for sample in DEFAULT_SAMPLES:
            p = Path(sample)
            if not p.exists():
                # Fallback: mira a ./data/ relatiu al fitxer actual.
                alt = Path(__file__).parent / "data" / Path(sample).name
                p = alt if alt.exists() else p
            if p.exists():
                # results.update(generate_plate_candidates(str(p), verbose=True))
                results.update(generate_plate_candidates(str(p), verbose=True))
            else:
                print(f"[INFO] Mostra no trobada: {sample}")

    print("\n=== RESUM ===")
    for name, cands in results.items():
        print(f"  {name}: {len(cands)} candidats")
        for c in cands:
            print(
                f"    bbox=({c['x1']},{c['y1']})-({c['x2']},{c['y2']}) "
                f"AR={c['aspect_ratio']:.2f} "
                f"area={c['area']}"
            )


if __name__ == "__main__":
    main()
