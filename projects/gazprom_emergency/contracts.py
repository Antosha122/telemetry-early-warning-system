"""Абстракции и контракты (Protocol/ABC) + реестры для расширяемости.

Решает замечание #8 senior review:
- ``ClassifierModelProtocol`` — контракт для любой модели-классификатора.
- ``OptimizerBuilder`` — контракт для фабрики оптимизаторов.
- ``FeatureScaler`` — контракт для scaler'ов (sklearn-совместимый).
- ``ModelRegistry`` / ``OptimizerRegistry`` — реестры на базе dict,
  позволяющие добавлять новые модели/оптимизаторы без правки ядра
  (Open/Closed Principle).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

# ============================================================
# Версия конфигурации чекпоинта (для обратной совместимости)
# ============================================================

CONFIG_VERSION = "2.0.0"

# ============================================================
# Protocols (структурная типизация — runtime-checkable)
# ============================================================


@runtime_checkable
class ClassifierModelProtocol(Protocol):
    """Контракт для модели бинарного классификатора.

    Любой ``nn.Module``-совместимый объект с ``forward``,
    ``parameters``, ``train/eval``, ``state_dict/load_state_dict``
    удовлетворяет этому протоколу автоматически.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def parameters(self) -> Any: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def load_state_dict(
        self, state_dict: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...


@runtime_checkable
class OptimizerBuilder(Protocol):
    """Контракт для фабрики оптимизаторов.

    Функция (или callable-объект), принимающая модель и параметры,
    возвращающая ``torch.optim.Optimizer``.
    """

    def __call__(
        self, model: nn.Module, **kwargs: Any
    ) -> torch.optim.Optimizer: ...


@runtime_checkable
class FeatureScaler(Protocol):
    """Контракт для scaler'а признаков (sklearn-совместимый).

    Требует только ``transform`` и ``n_features_in_``.
    """

    n_features_in_: int

    def transform(self, X: Any) -> Any: ...


# ============================================================
# Реестры (Registry Pattern)
# ============================================================


class ModelRegistry:
    """Реестр фабрик моделей.

    Позволяет регистрировать новые архитектуры без модификации ядра:

    .. code-block:: python

        @ModelRegistry.register("emergency_predictor")
        def build(model_cfg, **kwargs):
            return EmergencyPredictor(...)

    Использование::

        model = ModelRegistry.build("emergency_predictor", model_cfg=cfg)
    """

    _builders: dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        """Декоратор для регистрации фабрики модели."""

        def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            cls._builders[name.lower()] = builder
            return builder

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> nn.Module:
        """Создаёт модель по имени из реестра."""
        key = name.lower()
        if key not in cls._builders:
            available = ", ".join(sorted(cls._builders))
            raise ValueError(
                f"Unknown model type: {name!r}. "
                f"Available: [{available}]. "
                f"Register via @ModelRegistry.register(...)."
            )
        return cls._builders[key](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Возвращает список зарегистрированных имён моделей."""
        return sorted(cls._builders)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name.lower() in cls._builders

    @classmethod
    def _clear(cls) -> None:
        """Очищает реестр (только для тестов)."""
        cls._builders.clear()


class OptimizerRegistry:
    """Реестр фабрик оптимизаторов.

    Позволяет добавлять новые оптимизаторы декларативно:

    .. code-block:: python

        @OptimizerRegistry.register("adamw")
        def _build_adamw(model, lr=0.001, weight_decay=0.01, **_):
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    """

    _builders: dict[str, OptimizerBuilder] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[OptimizerBuilder], OptimizerBuilder]:
        """Декоратор для регистрации фабрики оптимизатора."""

        def decorator(builder: OptimizerBuilder) -> OptimizerBuilder:
            cls._builders[name.lower()] = builder
            return builder

        return decorator

    @classmethod
    def build(cls, name: str, model: nn.Module, **kwargs: Any) -> torch.optim.Optimizer:
        """Создаёт оптимизатор по имени из реестра."""
        key = name.lower()
        if key not in cls._builders:
            available = ", ".join(sorted(cls._builders))
            raise ValueError(
                f"Unknown optimizer: {name!r}. "
                f"Available: [{available}]. "
                f"Register via @OptimizerRegistry.register(...)."
            )
        return cls._builders[key](model, **kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Возвращает список зарегистрированных оптимизаторов."""
        return sorted(cls._builders)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name.lower() in cls._builders

    @classmethod
    def _clear(cls) -> None:
        """Очищает реестр (только для тестов)."""
        cls._builders.clear()