"""Единая архитектура модели прогноза аварий (PyTorch MLP)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class EmergencyPredictor(nn.Module):
    """MLP-классификатор аварий.

    Архитектура: input_dim -> hidden_dims -> 1 (logit).
    Каждый скрытый слой: Linear -> BatchNorm -> ReLU -> Dropout.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def build_model(input_dim: int, cfg: ModelConfig) -> EmergencyPredictor:
    """Создаёт модель из конфигурации."""
    return EmergencyPredictor(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )


def save_model(model: nn.Module, path: str) -> None:
    """Сохраняет веса модели."""
    torch.save(
        {
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"Model saved to {path}")


def load_model(path: str, input_dim: int, cfg: ModelConfig) -> EmergencyPredictor:
    """Загружает веса в модель."""
    model = build_model(input_dim, cfg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
