"""Тесты для contracts.py: Protocol'ы, реестры моделей и оптимизаторов."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from projects.gazprom_emergency.contracts import (
    ClassifierModelProtocol,
    ModelRegistry,
    OptimizerRegistry,
)
from projects.gazprom_emergency.config import Config, ModelConfig, TrainingConfig
from projects.gazprom_emergency.model import EmergencyPredictor
from projects.gazprom_emergency.optimizers import build_optimizer


# ============================================================
# Protocol: ClassifierModelProtocol
# ============================================================

class TestClassifierModelProtocol:
    """Тестирует Protocol для моделей-классификаторов."""

    def test_emergency_predictor_satisfies_protocol(self):
        """EmergencyPredictor удовлетворяет ClassifierModelProtocol (structural typing)."""
        model = EmergencyPredictor(input_dim=10, hidden_dims=[8], dropout=0.0)
        assert isinstance(model, ClassifierModelProtocol)

    def test_custom_module_satisfies_protocol(self):
        """Любой nn.Module с forward удовлетворяет протоколу."""

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x).squeeze(-1)

        model = CustomModel()
        assert isinstance(model, ClassifierModelProtocol)

    def test_non_module_does_not_satisfy(self):
        """Обычный объект НЕ удовлетворяет протоколу."""

        class NotAModel:
            pass

        obj = NotAModel()
        assert not isinstance(obj, ClassifierModelProtocol)


# ============================================================
# ModelRegistry
# ============================================================

class TestModelRegistry:
    """Тестирует реестр моделей (Registry Pattern)."""

    def test_emergency_predictor_is_registered(self):
        """Архитектура 'emergency_predictor' зарегистрирована по умолчанию."""
        assert ModelRegistry.is_registered("emergency_predictor")
        assert "emergency_predictor" in ModelRegistry.available()

    def test_build_known_model(self):
        """build создаёт модель по имени из реестра."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = ModelRegistry.build(
            "emergency_predictor", input_dim=10, model_cfg=cfg
        )
        assert isinstance(model, EmergencyPredictor)

    def test_build_unknown_model_raises(self):
        """Неизвестная архитектура вызывает ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelRegistry.build("nonexistent_model", input_dim=10, model_cfg=ModelConfig())

    def test_register_custom_model(self):
        """Можно зарегистрировать кастомную модель через декоратор."""

        @ModelRegistry.register("test_custom_model")
        def _build_custom(*, input_dim: int, model_cfg, **_):
            return nn.Linear(input_dim, 1)

        try:
            assert ModelRegistry.is_registered("test_custom_model")
            model = ModelRegistry.build(
                "test_custom_model", input_dim=5, model_cfg=ModelConfig()
            )
            assert isinstance(model, nn.Linear)
        finally:
            del ModelRegistry._builders["test_custom_model"]


# ============================================================
# OptimizerRegistry
# ============================================================

class TestOptimizerRegistry:
    """Тестирует реестр оптимизаторов (Registry Pattern)."""

    def test_predefined_optimizers_registered(self):
        """Adam, AdamW, RMSprop, SGD зарегистрированы по умолчанию."""
        for name in ["adam", "adamw", "rmsprop", "sgd"]:
            assert OptimizerRegistry.is_registered(name)

    def test_build_optimizer_from_config(self):
        """build_optimizer создаёт оптимизатор через реестр."""
        model = nn.Linear(10, 1)
        cfg = Config(training=TrainingConfig(optimizer="adam", learning_rate=0.01))
        opt = build_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_unknown_optimizer_raises(self):
        """Неизвестный оптимизатор вызывает ValueError (не молчаливый fallback)."""
        model = nn.Linear(10, 1)
        cfg = Config(training=TrainingConfig(optimizer="nonexistent"))
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model, cfg)

    def test_register_custom_optimizer(self):
        """Можно зарегистрировать кастомный оптимизатор."""

        @OptimizerRegistry.register("test_custom_opt")
        def _build_custom(model, lr=0.001, **_):
            return torch.optim.SGD(model.parameters(), lr=lr)

        try:
            model = nn.Linear(5, 1)
            opt = OptimizerRegistry.build("test_custom_opt", model, lr=0.1)
            assert isinstance(opt, torch.optim.SGD)
            assert opt.param_groups[0]["lr"] == 0.1
        finally:
            del OptimizerRegistry._builders["test_custom_opt"]