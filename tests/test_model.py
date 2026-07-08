"""Тесты для model.py: архитектура, save/load с метаданными, weights_only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from projects.gazprom_emergency.config import ModelConfig
from projects.gazprom_emergency.model import (
    EmergencyPredictor,
    build_model,
    load_model,
    save_model,
)


# ============================================================
# Архитектура модели
# ============================================================

class TestEmergencyPredictor:
    """Тестирует архитектуру MLP EmergencyPredictor."""

    def test_model_forward_shape(self):
        """Forward pass возвращает ожидаемую размерность."""
        model = EmergencyPredictor(input_dim=10, hidden_dims=[8, 4], dropout=0.0)
        X = torch.randn(5, 10)
        out = model(X)
        assert out.shape == (5,)  # squeeze(-1) убирает последнюю ось

    def test_build_model_from_config(self):
        """build_model создаёт модель с параметрами из конфига."""
        cfg = ModelConfig(hidden_dims=[16, 8], dropout=0.5)
        model = build_model(10, cfg)
        assert isinstance(model, EmergencyPredictor)

    def test_model_handles_batch_size_1(self):
        """Модель работает с batch_size=1 (важно для BatchNorm)."""
        model = EmergencyPredictor(input_dim=4, hidden_dims=[4], dropout=0.0)
        model.eval()  # BatchNorm в eval mode
        X = torch.randn(1, 4)
        out = model(X)
        assert out.shape == (1,)

    def test_model_output_is_logit(self):
        """Выход модели — logit (не ограничен [0, 1])."""
        model = EmergencyPredictor(input_dim=4, hidden_dims=[4], dropout=0.0)
        model.eval()
        X = torch.randn(10, 4)
        out = model(X)
        # Logits могут быть больше 1 или меньше 0
        assert (out.abs() > 0).any()


# ============================================================
# Save / Load с метаданными
# ============================================================

class TestSaveLoadModel:
    """Тестирует безопасное сохранение и загрузку модели."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Save → load восстанавливает модель с одинаковыми весами."""
        cfg = ModelConfig(hidden_dims=[8, 4], dropout=0.1)
        model = build_model(10, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=10, cfg=cfg)
        loaded = load_model(path)

        # Веса должны совпадать
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            torch.testing.assert_close(v1, v2)

    def test_load_without_explicit_input_dim(self, tmp_path: Path):
        """Загрузка без явного input_dim извлекает его из чекпоинта."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.0)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=5, cfg=cfg)
        # Загружаем без input_dim
        loaded = load_model(path)
        assert isinstance(loaded, EmergencyPredictor)

    def test_load_without_input_dim_in_checkpoint_raises(self, tmp_path: Path):
        """Загрузка без input_dim и без него в чекпоинте вызывает ValueError."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.0)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        # Сохраняем без метаданных
        torch.save({"state_dict": model.state_dict()}, path)

        with pytest.raises(ValueError, match="input_dim not provided"):
            load_model(path)

    def test_checkpoint_contains_metadata(self, tmp_path: Path):
        """Чекпоинт содержит архитектуру и метаданные."""
        cfg = ModelConfig(hidden_dims=[8, 4], dropout=0.2)
        model = build_model(10, cfg)
        path = str(tmp_path / "model.pth")
        feature_cols = [f"v_{i}" for i in range(10)]

        save_model(
            model,
            path,
            input_dim=10,
            cfg=cfg,
            feature_columns=feature_cols,
            extra={"optimizer": "adam", "seed": 42},
        )

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["architecture"] == "EmergencyPredictor"
        assert checkpoint["input_dim"] == 10
        assert checkpoint["hidden_dims"] == [8, 4]
        assert checkpoint["dropout"] == 0.2
        assert checkpoint["feature_columns"] == feature_cols
        assert checkpoint["n_features"] == 10
        assert checkpoint["config_version"] == "2.0.0"
        assert checkpoint["extra"]["optimizer"] == "adam"
        assert checkpoint["extra"]["seed"] == 42

    def test_save_model_with_feature_columns(self, tmp_path: Path):
        """save_model с feature_columns сохраняет их в чекпоинт."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.0)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=5, feature_columns=["a", "b", "c", "d", "e"])

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["feature_columns"] == ["a", "b", "c", "d", "e"]
        assert checkpoint["n_features"] == 5

    def test_loaded_model_is_in_eval_mode(self, tmp_path: Path):
        """После load_model модель находится в eval mode."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.0)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=5, cfg=cfg)
        loaded = load_model(path)
        assert not loaded.training  # eval mode