"""Фабрика оптимизаторов на базе реестра (Registry Pattern).

Решает замечание #8 senior review:
- ``build_optimizer`` использует реестр вместо жёсткого if/elif.
- Добавление нового оптимизатора = декоратор ``@OptimizerRegistry.register(...)``,
  без правки ядра.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .config import Config
from .contracts import OptimizerRegistry

logger = logging.getLogger(__name__)


# ============================================================
# Регистрация оптимизаторов (декларативно)
# ============================================================


@OptimizerRegistry.register("adam")
def _build_adam(
    model: nn.Module, *, lr: float = 0.001, weight_decay: float = 0.0001, **_: Any
) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@OptimizerRegistry.register("adamw")
def _build_adamw(
    model: nn.Module, *, lr: float = 0.001, weight_decay: float = 0.01, **_: Any
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@OptimizerRegistry.register("rmsprop")
def _build_rmsprop(
    model: nn.Module, *, lr: float = 0.001, **_: Any
) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(model.parameters(), lr=lr)


@OptimizerRegistry.register("sgd")
def _build_sgd(
    model: nn.Module,
    *,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    **_: Any,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


# ============================================================
# Публичный API
# ============================================================


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Создаёт оптимизатор по конфигурации (через реестр).

    Имя оптимизатора берётся из ``cfg.training.optimizer``.
    Параметры (lr, weight_decay) — из конфигурации.

    Неизвестный оптимизатор → ``ValueError`` (не молчаливый fallback),
    чтобы избежать скрытых багов.
    """
    name = cfg.training.optimizer
    if not OptimizerRegistry.is_registered(name):
        available = ", ".join(OptimizerRegistry.available())
        logger.error("Unknown optimizer %r. Available: [%s]", name, available)
        raise ValueError(
            f"Unknown optimizer: {name!r}. "
            f"Available: [{available}]. "
            f"Register via @OptimizerRegistry.register(...)."
        )

    optimizer = OptimizerRegistry.build(
        name,
        model,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    logger.info("Optimizer '%s' created (lr=%s)", name, cfg.training.learning_rate)
    return optimizer
