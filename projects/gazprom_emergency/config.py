"""Конфигурация пайплайна прогноза аварий (загрузка из YAML)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ===== dataclasses для типизированной конфигурации =====


@dataclass
class DataConfig:
    source_dir: str = ""
    opers_file: str = "opers.csv"
    stpa_file: str = "stpa.csv"
    processed_dir: str = ""
    chunk_size: int = 50000


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    dropout: float = 0.3
    save_path: str = ""


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 50
    batch_size: int = 256
    use_smote: bool = True
    smote_sampling_strategy: float = 0.5
    test_size: float = 0.2
    random_state: int = 42
    early_stopping_patience: int = 7


@dataclass
class PredictionConfig:
    threshold: float = 0.5
    horizon_hours: int = 3
    output_path: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


# ===== Утилиты для подстановки переменных окружения =====

_ENV_RE = re.compile(r"\$\{([^}^{]+)\}")


def _expand_env(value: str) -> str:
    """Подставляет значения ${VAR} из переменных окружения."""

    def _replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    return _ENV_RE.sub(_replace, value)


def _expand_recursive(obj: object) -> object:
    """Рекурсивно обходит dict/list и подставляет ${VAR} во всех строках."""
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_recursive(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


# ===== Основная функция загрузки конфигурации =====


def _section_dict(raw: dict, key: str) -> dict:
    """Достаёт секцию из dict и гарантирует возврат dict."""
    data = raw.get(key, {})
    return data if isinstance(data, dict) else {}


def _build_config(raw: dict) -> Config:
    """Собирает типизированный Config из «сырого» dict."""
    return Config(
        data=DataConfig(**_section_dict(raw, "data")),
        model=ModelConfig(**_section_dict(raw, "model")),
        training=TrainingConfig(**_section_dict(raw, "training")),
        prediction=PredictionConfig(**_section_dict(raw, "prediction")),
    )


def load_config(path: str | Path) -> Config:
    """Загружает конфигурацию из YAML-файла.

    Поддерживает подстановку переменных окружения в формате `${VAR}`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raw = {}

    expanded = _expand_recursive(raw)
    assert isinstance(expanded, dict)
    return _build_config(expanded)