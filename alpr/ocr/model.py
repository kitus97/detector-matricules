"""
alpr/ocr/model.py
=================
Arquitectura CharCNN per a classificació de caràcters 1×28×28.

Separat de train.py perquè infer.py i eval_real.py el puguin importar
directament sense haver de fer importlib sobre un script amb nom numèric.

Flux de dimensions:
  1×28×28
  → Conv(1→32, k=3, p=1) + BN + ReLU + MaxPool(2)  →  32×14×14
  → Conv(32→64, k=3, p=1) + BN + ReLU + MaxPool(2) →  64×7×7
  → Dropout(0.4)
  → Flatten                                         →  3136
  → Linear(3136→256) + ReLU
  → Dropout(0.3)
  → Linear(256→n_classes)  [logits, sense Softmax]
"""

import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """
    CNN compacte per a la classificació de caràcters alfanumèrics (36 classes: A–Z, 0–9).
    Entrada: tensor 1×28×28 normalitzat a [-1, 1].
    Sortida: logits crus (sense Softmax) de mida n_classes.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 — detecta contorns i traços bàsics
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 28 → 14

            # Bloc 2 — combina característiques de nivell mig
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 14 → 7

            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),         # CrossEntropyLoss ja inclou Softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
