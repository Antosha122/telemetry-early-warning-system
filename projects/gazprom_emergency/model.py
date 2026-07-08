"""Единая архитектура модели прогноза аварий (PyTorch MLP).

Решает замечания #7, #8 senior review:
- Веса сохраняются **с полными метаданными**: ``input_dim``, ``hidden_dims``,
  ``dropout``, ``feature_columns``, ``n_features``, ``config_version``.
- Модель и оптимизаторы регистрируются через реестр (Registry Pattern),
  что позволяет расширять архитектуры без правки ядра.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .config import ModelConfig
from .contracts import CONFIG_VERSION, ModelRegistry

logger = logging.getLogger(__name__)


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


# ============================================================
# Реестр моделей (Registry Pattern)
# ============================================================


@ModelRegistry.register("emergency_predictor")
def _build_emergency_predictor(
    *, input_dim: int, model_cfg: ModelConfig, **_: Any
) -> EmergencyPredictor:
    """Фабрика для EmergencyPredictor (зарегистрирована в реестре)."""
    return EmergencyPredictor(
        input_dim=input_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )


def build_model(input_dim: int, cfg: ModelConfig) -> EmergencyPredictor:
    """Создаёт модель из конфигурации (через реестр).

    По умолчанию используется архитектура ``emergency_predictor``.
    Для добавления новой архитектуры — зарегистрируйте её через
    ``@ModelRegistry.register(...)`` и укажите имя в конфиге.
    """
    architecture = getattr(cfg, "architecture", "emergency_predictor")
    model = ModelRegistry.build(
        architecture,
        input_dim=input_dim,
        model_cfg=cfg,
    )
    return model  # type: ignore[return-value]


# ============================================================
# Сериализация с метаданными (безопасная загрузка)
# ============================================================


def save_model(
    model: nn.Module,
    path: str,
    *,
    input_dim: int | None = None,
    cfg: ModelConfig | None = None,
    feature_columns: list[str] | None = None,
    n_features: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Сохраняет веса модели и метаданные архитектуры в чекпоинт.

    Args:
        model: обученная модель.
        path: путь к файлу (.pth).
        input_dim: размерность входа (сохраняется в чекпоинт).
        cfg: конфигурация модели (сохраняется в чекпоинт).
        feature_columns: упорядоченный список имён признаков (для инференса).
        n_features: число признаков (= ``len(feature_columns)``).
        extra: дополнительные метаданные (optimizer, seed, training config).
    """
    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "architecture": "EmergencyPredictor",
        "config_version": CONFIG_VERSION,
    }
    if input_dim is not None:
        checkpoint["input_dim"] = int(input_dim)
    if cfg is not None:
        checkpoint["hidden_dims"] = list(cfg.hidden_dims)
        checkpoint["dropout"] = float(cfg.dropout)
    if feature_columns is not None:
        checkpoint["feature_columns"] = list(feature_columns)
        checkpoint["n_features"] = len(feature_columns)
    elif n_features is not None:
        checkpoint["n_features"] = int(n_features)
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info("Model saved to %s (config_version=%s)", path, CONFIG_VERSION)


def load_model(
    path: str,
    input_dim: int | None = None,
    cfg: ModelConfig | None = None,
) -> EmergencyPredictor:
    """Загружает веса в модель (безопасно, weights_only=True).

    Если архитектура сохранена в чекпоинте, ``input_dim`` и ``cfg`` могут быть
    извлечены из него; иначе требуется явная передача.

    Args:
        path: путь к файлу (.pth).
        input_dim: размерность входа (если None — берётся из чекпоинта).
        cfg: конфигурация модели (если None — берётся из чекпоинта).

    Raises:
        ValueError: если ``input_dim`` не указан и отсутствует в чекпоинте.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # Пытаемся извлечь архитектуру из чекпоинта
    if input_dim is None:
        input_dim = checkpoint.get("input_dim")
        if input_dim is None:
            # Fallback на n_features для обратной совместимости
            input_dim = checkpoint.get("n_features")
        if input_dim is None:
            raise ValueError(
                "input_dim not provided and not found in checkpoint. "
                "Cannot reconstruct model architecture."
            )

    if cfg is None:
        hidden_dims = checkpoint.get("hidden_dims", [256, 128, 64, 32])
        dropout = checkpoint.get("dropout", 0.3)
        cfg = ModelConfig(hidden_dims=hidden_dims, dropout=dropout)

    model = build_model(input_dim, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(
        "Model loaded from %s (input_dim=%d, hidden_dims=%s, config_version=%s)",
        path,
        input_dim,
        cfg.hidden_dims,
        checkpoint.get("config_version", "unknown"),
    )
    return model


def load_checkpoint_metadata(path: str) -> dict[str, Any]:
    """Загружает только метаданные чекпоинта (без весов).

    Полезно для инференса: проверить ``feature_columns``, ``input_dim``,
    ``n_features`` без загрузки полной модели.

    Args:
        path: путь к файлу (.pth).

    Returns:
        dict с метаданными чекпоинта.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Возвращаем всё, кроме state_dict (он тяжёлый)
    return {k: v for k, v in checkpoint.items() if k != "state_dict"}