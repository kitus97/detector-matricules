"""
03_train_cnn.py
===============
Pas 7 del pla d'entrenament OCR.

Entrena un CNN compacte (CharCNN) sobre el dataset sintètic generat als
passos anteriors. El dataset ha d'estar a data/synthetic/ en format
torchvision.ImageFolder (train/ i val/) amb imatges 28×28 px binàries.

Prerequisit: executar primer 01_render_fonts.py i 02_augment.py.

Ús
--
    python 03_train_cnn.py
    python 03_train_cnn.py --epochs 30 --batch_size 64 --lr 1e-3 --seed 42
    python 03_train_cnn.py --epochs 2   # verificació ràpida
"""

import argparse
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


# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Càrrega de dades
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Construeix els DataLoaders de train i val des de data_dir/train i data_dir/val.

    Transforma les imatges a tensors 1×28×28 normalitzats a [-1, 1].
    No s'aplica cap augmentation addicional aquí: ja es va fer al pas 4.

    Retorna (train_loader, val_loader, class_to_idx).
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # força canal únic per si el PNG s'obre com RGB
        transforms.ToTensor(),                         # [0,255] PIL → [0,1] tensor (1×H×W)
        transforms.Normalize((0.5,), (0.5,)),          # [0,1] → [-1,1]
    ])

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise FileNotFoundError(
            f"No s'ha trobat el dataset a '{train_dir}'.\n"
            "Executa primer 01_render_fonts.py i 02_augment.py."
        )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=transform)

    # Verifica que el class_to_idx del ImageFolder coincideix amb el JSON guardat
    json_path = data_dir / "class_to_idx.json"
    if json_path.exists():
        saved_map = json.loads(json_path.read_text())
        if saved_map != train_dataset.class_to_idx:
            raise ValueError(
                "El class_to_idx del ImageFolder no coincideix amb class_to_idx.json.\n"
                "Regenera el dataset amb 02_augment.py."
            )

    # num_workers=0 evita problemes amb el fork de multiprocessing a macOS
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    log.info(f"Train: {len(train_dataset)} imatges, {len(train_dataset.classes)} classes")
    log.info(f"Val  : {len(val_dataset)} imatges")

    return train_loader, val_loader, train_dataset.class_to_idx


# ══════════════════════════════════════════════════════════════════════════════
# Arquitectura del model
# ══════════════════════════════════════════════════════════════════════════════

class CharCNN(nn.Module):
    """
    CNN compacte per a classificació de caràcters en imatges binàries 1×28×28.

    Flux de dimensions:
      1 × 28 × 28
      → Conv(1→32, k=3, p=1) + BN + ReLU + MaxPool(2)  →  32 × 14 × 14
      → Conv(32→64, k=3, p=1) + BN + ReLU + MaxPool(2) →  64 × 7 × 7
      → Dropout(0.4)
      → Flatten                                         →  3136
      → Linear(3136→256) + ReLU
      → Dropout(0.3)
      → Linear(256→n_classes)   [logits crus, sense Softmax]
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 — detecta contorns i traços bàsics
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # padding=1 preserva W×H
            nn.BatchNorm2d(32),                           # estabilitza activacions batch a batch
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28 → 14

            # Bloc 2 — combina característiques de nivell mig
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14 → 7

            nn.Dropout(0.4),                              # regularització: crític perquè entrenem amb sintètics
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),                    # logits — CrossEntropyLoss ja inclou Softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# Bucle d'entrenament i avaluació
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Fa una passada completa per train. Retorna (loss_mitja, accuracy)."""
    model.train()
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

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
    """Avalua el model sense gradient. Retorna (loss_mitja, accuracy)."""
    model.eval()
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

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
# Corbes d'entrenament
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(
    history: dict[str, list[float]],
    out_path: Path,
    best_epoch: int,
) -> None:
    """Guarda una figura amb les corbes de loss i accuracy (train vs val)."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, label=f"millor (ep. {best_epoch})")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    plt.suptitle("Corbes d'entrenament CharCNN", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    log.info(f"Corbes guardades a: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Punt d'entrada principal
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena CharCNN per a l'OCR de matrícules."
    )
    parser.add_argument("--data_dir",   default="data/synthetic",
                        help="Directori del dataset (default: data/synthetic)")
    parser.add_argument("--out_dir",    default="models",
                        help="Directori de sortida dels checkpoints (default: models)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Epochs sense millora per a l'early stopping (default: 5)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # ── Reproductibilitat ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # ── Dades ───────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_dataloaders(data_dir, args.batch_size)
    n_classes = len(class_to_idx)

    # ── Model, loss, optimitzador ────────────────────────────────────────────
    model     = CharCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log.info(f"Model: CharCNN  ({n_classes} classes)")
    log.info(f"Paràmetres: {sum(p.numel() for p in model.parameters()):,}")
    log.info("-" * 60)
    log.info(f"{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'*':>2}")
    log.info("-" * 60)

    # ── Bucle d'entrenament ──────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc     = -1.0
    best_epoch       = 1
    patience_counter = 0
    ckpt_path        = out_dir / "char_cnn_best.pth"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val_acc
        marker   = "✓" if improved else ""

        log.info(
            f"{epoch:>5}  {tr_loss:>7.4f}  {tr_acc:>6.2%}  "
            f"{va_loss:>7.4f}  {va_acc:>6.2%}  {marker}"
        )

        if improved:
            best_val_acc     = va_acc
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "val_acc":       best_val_acc,
                "n_classes":     n_classes,
                "class_to_idx":  class_to_idx,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping a l'epoch {epoch} ({args.patience} epochs sense millora)")
                break

    # ── Resum final ─────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info(f"Millor val accuracy: {best_val_acc:.2%}  (epoch {best_epoch})")
    log.info(f"Checkpoint guardat : {ckpt_path}")

    # ── Corbes ──────────────────────────────────────────────────────────────
    plot_curves(history, out_dir / "training_curves.png", best_epoch)


if __name__ == "__main__":
    main()
