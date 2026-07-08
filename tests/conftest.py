"""Общие fixtures для тестов gazprom_emergency."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Добавляем корень проекта в sys.path для импорта projects.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию для артефактов."""
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию для данных."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_opers_df():
    """Возвращает миниатюрный DataFrame opers для тестов merge."""
    import polars as pl

    return pl.DataFrame(
        {
            "batch_time": [
                "2023-01-01 00:00:00",
                "2023-01-01 03:00:00",
                "2023-01-01 06:00:00",
                "2023-01-01 09:00:00",
                "2023-01-01 12:00:00",
            ],
            "is_emergency": [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_stpa_df():
    """Возвращает миниатюрный DataFrame stpa для тестов merge (5 признаков)."""
    import polars as pl

    n_rows = 5
    data = {
        "batch_time": [
            "2023-01-01 00:00:00",
            "2023-01-01 03:00:00",
            "2023-01-01 06:00:00",
            "2023-01-01 09:00:00",
            "2023-01-01 12:00:00",
        ],
    }
    for i in range(5):
        data[f"v_{i}"] = np.arange(n_rows, dtype=np.float32) + i
    return pl.DataFrame(data)


@pytest.fixture
def sample_config_dict():
    """Возвращает «сырой» dict конфигурации для тестов."""
    return {
        "data": {
            "source_dir": "/tmp/data",
            "opers_file": "opers.csv",
            "stpa_file": "stpa.csv",
            "processed_dir": "/tmp/processed",
            "chunk_size": 1000,
        },
        "model": {
            "hidden_dims": [32, 16],
            "dropout": 0.2,
            "save_path": "/tmp/model.pth",
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 32,
            "use_smote": False,
            "test_size": 0.2,
            "split_strategy": "chronological",
            "random_state": 42,
            "seed": 42,
            "early_stopping_patience": 3,
        },
        "prediction": {
            "threshold": 0.5,
            "horizon_hours": 3,
            "output_path": "/tmp/predictions.csv",
        },
    }


@pytest.fixture
def sample_config_yaml(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Создаёт временный config.yaml для тестов."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Очищает переменные окружения до и после каждого теста."""
    keys = ["DATA_DIR", "ARTIFACTS_DIR"]
    old = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in old.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]