from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

@dataclass
class PlateDetection:
    """Retorn de la fase 1 i 2 (Detector + Alineació)."""
    contour: np.ndarray             # Els 4 vèrtexs detectats a la imatge original
    aligned_image: np.ndarray       # La imatge de la matrícula ja rectangular (Warped)
    binary_image: np.ndarray        # La matrícula binaritzada (blanca i negra) llesta per segmentar

@dataclass
class CharacterSegment:
    """Retorn de la fase 3 (Segmentació). Un objecte per cada lletra/número."""
    image: np.ndarray               # La sub-imatge del caràcter aïllat (p.ex. 20x30 píxels)
    bounding_box: tuple             # (x, y, w, h) Dins de la matrícula alineada (útil per dibuixar-ho després)
    position_index: int             # L'ordre del caràcter (0, 1, 2...) d'esquerra a dreta

@dataclass
class OCRResult:
    """Retorn de la fase 4 (Reconeixement)."""
    text: str                       # El text final reconegut (ex: "1234ABC")
    confidence_scores: List[float]  # Nivell de "seguretat" de l'OCR per cada lletra

@dataclass
class LicensePlate:
    """L'objecte mestre que agrupa tota la informació d'una matrícula processada."""
    original_image: np.ndarray              # La imatge sencera del cotxe
    detection: Optional[PlateDetection] = None
    characters: List[CharacterSegment] = field(default_factory=list)
    ocr: Optional[OCRResult] = None
    
    @property
    def is_fully_processed(self) -> bool:
        """Una propietat útil per saber si la pipeline ha acabat amb èxit."""
        return self.ocr is not None and len(self.ocr.text) > 0