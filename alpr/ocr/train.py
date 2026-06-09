"""
alpr/ocr/train.py
=================
Fase 3 — entrenament de CharCNN sobre el dataset sintètic.

Llegeix data/synthetic/{train,val}/ (format ImageFolder, 28×28 px, binàries).
Desa el millor checkpoint a models/char_cnn_best.pth.
"""

import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from alpr import config
from .model import CharCNN

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Càrrega de dades
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    data_dir: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Construeix DataLoaders de train i val des de data_dir/train i data_dir/val.
    Retorna (train_loader, val_loader, class_to_idx).
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise FileNotFoundError(
            f"No s'ha trobat el dataset a '{train_dir}'.\n"
            "Executa primer scripts/build_ocr_dataset.py."
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=transform)

    # Verifica coherència amb class_to_idx.json
    json_path = data_dir / "class_to_idx.json"
    if json_path.exists():
        saved = json.loads(json_path.read_text())
        if saved != train_ds.class_to_idx:
            raise ValueError(
                "class_to_idx del ImageFolder no coincideix amb class_to_idx.json.\n"
                "Regenera el dataset."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    log.info(f"Train: {len(train_ds)} imatges, {len(train_ds.classes)} classes")
    log.info(f"Val  : {len(val_ds)} imatges")

    return train_loader, val_loader, train_ds.class_to_idx


# ══════════════════════════════════════════════════════════════════════════════
# Bucles d'entrenament i avaluació
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Una passada completa de train. Retorna (loss_mitja, accuracy)."""
    model.train()
    total_loss = 0.0
    n_correct = n_total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_correct  += (logits.argmax(dim=1) == labels).sum().item()
        n_total    += imgs.size(0)

    return total_loss / n_total, n_correct / n_total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Avaluació sense gradient. Retorna (loss_mitja, accuracy)."""
    model.eval()
    total_loss = 0.0
    n_correct = n_total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n_correct  += (logits.argmax(dim=1) == labels).sum().item()
            n_total    += imgs.size(0)

    return total_loss / n_total, n_correct / n_total


# ══════════════════════════════════════════════════════════════════════════════
# Corbes
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(
    history: dict[str, list[float]],
    out_path: Path,
    best_epoch: int,
) -> None:
    """Guarda figura amb les corbes de loss i accuracy (train vs val)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8,
                    label=f"millor (ep. {best_epoch})")
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1); ax_acc.legend()

    plt.suptitle("Corbes d'entrenament CharCNN", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    log.info(f"Corbes guardades a: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Funció principal d'entrenament
# ══════════════════════════════════════════════════════════════════════════════

def train(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> Path:
    """
    Entrena CharCNN i desa el millor checkpoint.
    Retorna la ruta del checkpoint guardat.
    """
    if data_dir is None:
        data_dir = config.DATA_SYNTHETIC_DIR
    if out_dir is None:
        out_dir = config.MODELS_DIR

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_dataloaders(data_dir, batch_size)
    n_classes = len(class_to_idx)

    model     = CharCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log.info(f"Model: CharCNN ({n_classes} classes)")
    log.info(f"Paràmetres: {sum(p.numel() for p in model.parameters()):,}")

    history          = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc     = -1.0
    best_epoch       = 1
    patience_counter = 0
    ckpt_path        = out_dir / "char_cnn_best.pth"

    log.info("-" * 60)
    log.info(f"{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'*':>2}")
    log.info("-" * 60)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val_acc
        log.info(f"{epoch:>5}  {tr_loss:>7.4f}  {tr_acc:>6.2%}  "
                 f"{va_loss:>7.4f}  {va_acc:>6.2%}  {'✓' if improved else ''}")

        if improved:
            best_val_acc = va_acc; best_epoch = epoch; patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      best_val_acc,
                "n_classes":    n_classes,
                "class_to_idx": class_to_idx,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping a l'epoch {epoch}")
                break

    log.info(f"Millor val accuracy: {best_val_acc:.2%}  (epoch {best_epoch})")
    log.info(f"Checkpoint: {ckpt_path}")

    plot_curves(history, out_dir / "training_curves.png", best_epoch)
    return ckpt_path
