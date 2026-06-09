"""
config.py
=========
Única font d'hiperparàmetres i rutes del projecte ALPR.
Cap mòdul ha de tenir constants màgiques pròpies: tot passa per aquí.
"""

from pathlib import Path

# ── Rutes ──────────────────────────────────────────────────────────────────────
# Ancorades a l'arrel del projecte (alpr/config.py → parents[1]), no al cwd, perquè
# el pipeline funcioni des de qualsevol directori d'execució.
ROOT_DIR           = Path(__file__).resolve().parents[1]
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_CHARS_DIR     = ROOT_DIR / "data" / "chars"
DATA_SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
MODELS_DIR         = ROOT_DIR / "models"
MODEL_CNN_PATH     = MODELS_DIR / "char_cnn_best.pth"
OUTPUT_DIR         = ROOT_DIR / "output"
FONTS_DIR          = ROOT_DIR / "resources" / "fonts"

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Fase 1 — Detector ──────────────────────────────────────────────────────────
# Preprocessament
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)
GAUSS_KSIZE      = (5, 5)
GAUSS_SIGMA      = 1.0

# Morfologia  (mides pensades per a imatges ~400–600 px d'amplada;
#              augmentar per a resolucions superiors)
CLOSE_KERNEL = (15, 3)
OPEN_KERNEL  = (5, 5)

# Filtre de forma
AREA_RATIO_MIN   = 0.001   # 0.1 % de la imatge
AREA_RATIO_MAX   = 0.10    # 10 % de la imatge
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 9.0
EXTENT_MIN       = 0.25
MIN_WIDTH        = 40
MIN_HEIGHT       = 10

# ── Fase 2 — Segmentador ───────────────────────────────────────────────────────
NORM_H              = 96        # alçada de normalització del crop abans de detectar
                               # (D9/ADR-002: les plaques del dataset són de 18–46 px i
                               # blockSize=31 falla sense escalar a una alçada de treball comuna)
ADAPT_BLOCK         = 31        # blockSize per a adaptiveThreshold
ADAPT_C             = 15        # constant C per a adaptiveThreshold
MORPH_CLOSE_KERNEL  = (1, 3)    # consolida traços fins (p. ex. 'I') sense fusionar veïns

DESKEW_MIN_CANDIDATS = 3        # mínim de centroides per estimar l'angle (ADR-003)
DESKEW_ANGLE_MAX     = 15.0     # graus: si |angle| supera, no rota

H_CHAR_MIN_REL  = 0.30          # h_min = H_CHAR_MIN_REL · h_placa (filtre groller)
H_CHAR_MAX_REL  = 1.05          # h_max = H_CHAR_MAX_REL · h_placa
AR_MIN          = 0.05          # aspect ratio mínim (cobreix la 'I'/'1', AR real 0.05–0.12)
AR_MAX          = 1.10          # aspect ratio màxim (per sobre sol ser soroll o fusions)
AREA_MIN_REL    = 0.015         # àrea mínima del blob / h_placa² (elimina speckle)

H_MED_TOL       = 0.30          # banda de la fila dominant: [1±H_MED_TOL]·mediana d'alçades
ROW_STD_H_MAX   = 0.20          # std_h / h_med màxima (uniformitat)
ROW_STD_CY_MAX  = 0.25          # std_cy / h_med màxima (alineació baseline)
WIDTH_OCC_MIN   = 0.30          # cobertura horitzontal mínima del crop

N_CHARS_MIN     = 5             # rang de caràcters vàlids per a una matrícula
N_CHARS_MAX     = 9             # (matrícules EU del dataset)
IOU_OVERLAP     = 0.50          # llindar de deduplicació IoU

CANVAS_SIZE     = 64            # mida del llenç per als renders de font
MARGIN          = 6             # marge interior al llenç (píxels)
OUTPUT_SIZE     = 28            # mida de sortida del caràcter per a la CNN

FORMAT_SORTIDA  = "vj"          # "vj" (tight crop + resize) | "regla-or" (llenç 64×64)

# Projecció vertical (mètode de segmentació ALTERNATIU, opcional)
PROJ_SMOOTH_WIN = 9             # finestra de la mitjana mòbil del histograma de columnes
PROJ_INK_FRAC   = 0.12          # fracció del màxim per considerar una columna "amb tinta"

# ── Fase 3 — OCR ───────────────────────────────────────────────────────────────
CHARS          = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # 36 classes (A–Z + 0–9)
OCR_INPUT_SIZE = 28
CONF_THRESHOLD = 0.5            # filtre de baixa confiança en inferència
