"""Тесты для модуля config.py: парсинг, env-var подстановка, edge-cases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from projects.gazprom_emergency.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)


# ============================================================
# Парсинг базовой конфигурации
# ============================================================

class TestLoadConfig:
    """Тестирует load_config() — загрузку YAML в типизированный Config."""

    def test_load_valid_config(self, sample_config_yaml: Path):
        """Корректный YAML загружается без ошибок."""
        cfg = load_config(sample_config_yaml)
        assert isinstance(cfg, Config)

    def test_config_has_all_sections(self, sample_config_yaml: Path):
        """Все четыре секции (data, model, training, prediction) присутствуют."""
        cfg = load_config(sample_config_yaml)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_data_values_parsed(self, sample_config_yaml: Path):
        """Значения data-секции корректно парсятся."""
        cfg = load_config(sample_config_yaml)
        assert cfg.data.opers_file == "opers.csv"
        assert cfg.data.stpa_file == "stpa.csv"
        assert cfg.data.chunk_size == 1000

    def test_model_values_parsed(self, sample_config_yaml: Path):
        """Значения model-секции корректно парсятся."""
        cfg = load_config(sample_config_yaml)
        assert cfg.model.hidden_dims == [32, 16]
        assert cfg.model.dropout == 0.2

    def test_training_values_parsed(self, sample_config_yaml: Path):
        """Значения training-секции корректно парсятся."""
        cfg = load_config(sample_config_yaml)
        assert cfg.training.optimizer == "adam"
        assert cfg.training.learning_rate == 0.001
        assert cfg.training.batch_size == 32
        assert cfg.training.split_strategy == "chronological"
        assert cfg.training.seed == 42


# ============================================================
# Подстановка переменных окружения
# ============================================================

class TestEnvVarSubstitution:
    """Тестирует подстановку ${VAR} из переменных окружения."""

    def test_env_var_substitution(self, tmp_path: Path):
        """${VAR} заменяется на значение переменной окружения."""
        os.environ["TEST_DATA_DIR"] = "/custom/data/path"

        config_data = {"data": {"source_dir": "${TEST_DATA_DIR}"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.data.source_dir == "/custom/data/path"

    def test_env_var_not_set_keeps_original(self, tmp_path: Path):
        """Если переменная не задана, ${VAR} остаётся как есть."""
        if "NONEXISTENT_VAR_12345" in os.environ:
            del os.environ["NONEXISTENT_VAR_12345"]

        config_data = {"data": {"source_dir": "${NONEXISTENT_VAR_12345}"}}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.data.source_dir == "${NONEXISTENT_VAR_12345}"

    def test_env_var_in_multiple_places(self, tmp_path: Path):
        """${VAR} подставляется во всех строках рекурсивно."""
        os.environ["BASE_DIR"] = "/base"

        config_data = {
            "data": {
                "source_dir": "${BASE_DIR}/data",
                "processed_dir": "${BASE_DIR}/processed",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.data.source_dir == "/base/data"
        assert cfg.data.processed_dir == "/base/processed"


# ============================================================
# Edge-cases
# ============================================================

class TestEdgeCases:
    """Тестирует граничные случаи и обработку ошибок."""

    def test_file_not_found(self):
        """Несуществующий файл вызывает FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_yaml(self, tmp_path: Path):
        """Пустой YAML возвращает Config со значениями по умолчанию."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        cfg = load_config(config_path)
        assert isinstance(cfg, Config)
        assert cfg.data.chunk_size == 50000  # значение по умолчанию

    def test_partial_config(self, tmp_path: Path):
        """Конфиг с одной секцией загружается, остальные — по умолчанию."""
        config_data = {"data": {"chunk_size": 999}}
        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.data.chunk_size == 999
        # model должна быть по умолчанию
        assert cfg.model.hidden_dims == [256, 128, 64, 32]

    def test_non_dict_yaml(self, tmp_path: Path):
        """YAML со скалярным значением не падает, возвращает дефолты."""
        config_path = tmp_path / "scalar.yaml"
        config_path.write_text("just_a_string")

        cfg = load_config(config_path)
        assert isinstance(cfg, Config)