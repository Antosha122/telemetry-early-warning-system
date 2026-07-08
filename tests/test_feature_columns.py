"""Тесты для feature_columns и checkpoint metadata (решение замечания #7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.gazprom_emergency.config import ModelConfig
from projects.gazprom_emergency.data import (
    load_feature_columns,
    save_feature_columns,
)
from projects.gazprom_emergency.model import (
    build_model,
    load_checkpoint_metadata,
    save_model,
)


# ============================================================
# Feature columns: save/load (JSON)
# ============================================================

class TestFeatureColumns:
    """Тестирует сохранение/загрузку списка признаков в JSON."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """save -> load восстанавливает идентичный список."""
        cols = ["v_0", "v_1", "v_2", "v_3"]
        path = tmp_path / "feature_columns.json"

        save_feature_columns(cols, path)
        loaded = load_feature_columns(path)

        assert loaded == cols

    def test_save_creates_parent_dir(self, tmp_path: Path):
        """save создаёт родительские директории."""
        path = tmp_path / "deep" / "nested" / "feature_columns.json"
        save_feature_columns(["v_0"], path)
        assert path.exists()

    def test_json_contains_n_features(self, tmp_path: Path):
        """JSON содержит поле n_features."""
        cols = ["v_0", "v_1", "v_2"]
        path = tmp_path / "feature_columns.json"
        save_feature_columns(cols, path)

        with open(path) as f:
            data = json.load(f)
        assert data["n_features"] == 3
        assert data["feature_columns"] == cols

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """Загрузка несуществующего файла вызывает FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Feature columns file not found"):
            load_feature_columns(tmp_path / "nonexistent.json")

    def test_preserves_order(self, tmp_path: Path):
        """Порядок признаков сохраняется."""
        cols = ["v_100", "v_0", "v_50", "v_25"]
        path = tmp_path / "feature_columns.json"
        save_feature_columns(cols, path)
        loaded = load_feature_columns(path)
        assert loaded == cols  # Порядок не должен меняться


# ============================================================
# Checkpoint metadata: feature_columns + config_version
# ============================================================

class TestCheckpointMetadata:
    """Тестирует сохранение метаданных в чекпоинте."""

    def test_checkpoint_contains_feature_columns(self, tmp_path: Path):
        """Чекпоинт содержит feature_columns (решение замечания #7)."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = build_model(10, cfg)
        path = str(tmp_path / "model.pth")
        feature_cols = [f"v_{i}" for i in range(10)]

        save_model(model, path, input_dim=10, cfg=cfg, feature_columns=feature_cols)

        meta = load_checkpoint_metadata(path)
        assert meta["feature_columns"] == feature_cols
        assert meta["n_features"] == 10

    def test_checkpoint_contains_config_version(self, tmp_path: Path):
        """Чекпоинт содержит версию конфигурации."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=5, cfg=cfg)

        meta = load_checkpoint_metadata(path)
        assert "config_version" in meta
        assert meta["config_version"] == "2.0.0"

    def test_checkpoint_metadata_excludes_state_dict(self, tmp_path: Path):
        """load_checkpoint_metadata не возвращает state_dict (он тяжёлый)."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(model, path, input_dim=5, cfg=cfg)
        meta = load_checkpoint_metadata(path)

        assert "state_dict" not in meta
        assert "input_dim" in meta

    def test_checkpoint_without_feature_columns(self, tmp_path: Path):
        """Чекпоинт без feature_columns (обратная совместимость)."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        # Сохраняем без feature_columns
        save_model(model, path, input_dim=5, cfg=cfg)

        meta = load_checkpoint_metadata(path)
        assert "feature_columns" not in meta
        assert meta["input_dim"] == 5

    def test_checkpoint_contains_extra_metadata(self, tmp_path: Path):
        """Чекпоинт содержит extra-метаданные (optimizer, seed и т.д.)."""
        cfg = ModelConfig(hidden_dims=[8], dropout=0.1)
        model = build_model(5, cfg)
        path = str(tmp_path / "model.pth")

        save_model(
            model,
            path,
            input_dim=5,
            cfg=cfg,
            extra={"optimizer": "adam", "seed": 42, "split_strategy": "chronological"},
        )

        meta = load_checkpoint_metadata(path)
        assert meta["extra"]["optimizer"] == "adam"
        assert meta["extra"]["seed"] == 42
