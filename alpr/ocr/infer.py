"""
alpr/ocr/infer.py
=================
Fase 3 — inferència: càrrega del checkpoint i predicció de caràcters.

API pública:
  load_model(model_path)           -> (model, class_to_idx, idx_to_class, device)
  predict(char_img, model, ...)    -> (lletra, confiança)   accepta np.ndarray 28×28
  predict_char(model, img_path, …) -> (idx, confiança)      per a fitxers (compat. 04)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from alpr import config
from .model import CharCNN

log = logging.getLogger(__name__)

# Transform idèntic al de l'entrenament
_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.OCR_INPUT_SIZE, config.OCR_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# ══════════════════════════════════════════════════════════════════════════════
# Càrrega del model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_path: Path | None = None,
) -> tuple[nn.Module, dict[str, int], dict[int, str], torch.device]:
    """
    Carrega el checkpoint i reconstrueix CharCNN.
    Retorna (model, class_to_idx, idx_to_class, device).
    """
    if model_path is None:
        model_path = config.MODEL_CNN_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"No s'ha trobat el checkpoint a '{model_path}'.\n"
            "Executa primer scripts/train_ocr.py."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt         = torch.load(str(model_path), map_location=device, weights_only=False)
    n_classes    = ckpt["n_classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CharCNN(n_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    val_acc = ckpt.get("val_acc")
    if val_acc is not None:
        log.info(f"Model carregat: {n_classes} classes | val_acc = {val_acc:.2%}")
    else:
        log.info(f"Model carregat: {n_classes} classes")

    return model, class_to_idx, idx_to_class, device


# ══════════════════════════════════════════════════════════════════════════════
# Inferència
# ══════════════════════════════════════════════════════════════════════════════

def _to_tensor(img: np.ndarray | Image.Image) -> torch.Tensor:
    """Converteix np.ndarray uint8 o PIL Image al tensor esperat pel model."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return _TRANSFORM(img).unsqueeze(0)


def predict(
    char_img: np.ndarray,
    model: nn.Module,
    idx_to_class: dict[int, str],
    device: torch.device,
    threshold: float | None = None,
) -> tuple[str, float]:
    """
    Prediu el caràcter d'una imatge np.ndarray 28×28 binària.

    Retorna (lletra, confiança_softmax).
    Si confiança < threshold retorna ('?', confiança).
    """
    if threshold is None:
        threshold = config.CONF_THRESHOLD

    tensor = _to_tensor(char_img).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()

    idx  = int(probs.argmax().item())
    conf = float(probs[idx].item())
    char = idx_to_class.get(idx, "?")

    if conf < threshold:
        return "?", conf
    return char, conf


def predict_char(
    model: nn.Module,
    img_path: Path,
    device: torch.device,
) -> tuple[int, float]:
    """
    Prediu el caràcter d'un fitxer d'imatge.
    Retorna (idx_predit, confiança_softmax).
    Compatible amb la firma original de 04_evaluate_real.py.
    """
    img    = Image.open(img_path).convert("L")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    idx  = int(probs.argmax().item())
    conf = float(probs[idx].item())
    return idx, conf


def filter_by_confidence(
    preds: list[dict],
    threshold: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Divideix prediccions en retingudes (conf >= threshold) i descartades.
    Cada element de preds ha de tenir la clau 'confidence'.
    """
    if threshold is None:
        threshold = config.CONF_THRESHOLD
    retained  = [p for p in preds if p["confidence"] >= threshold]
    discarded = [p for p in preds if p["confidence"] <  threshold]
    return retained, discarded
