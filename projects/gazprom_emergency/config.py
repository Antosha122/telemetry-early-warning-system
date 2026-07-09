"""Конфигурация пайплайна прогноза аварий (загрузка из YAML).

Решение замечаний senior review #11–#17:
- ``ValidationConfig`` — параметры валидации входных данных (#11).
- ``FeatureEngineeringConfig`` — обработка пропусков, временные признаки,
  лаги, скользящие статистики, PCA (#12).
- ``ExperimentConfig`` — сравнение моделей, K-Fold CV, поиск гиперпараметров (#13).
- ``ThresholdConfig`` — оптимизация порога, калибровка вероятностей (#14).
- ``MlflowConfig`` — трекинг экспериментов (#15).
- ``CostMatrixConfig`` — стоимости ошибок для cost-sensitive learning (#17).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ===== dataclasses для типизированной конфигурации =====


@dataclass
class DataConfig:
    source_dir: str = ""
    opers_file: str = "opers.csv"
    stpa_file: str = "stpa.csv"
    processed_dir: str = ""
    chunk_size: int = 50000
    # Разделители CSV (opers — запятая, stpa — точка с запятой)
    opers_separator: str = ","
    stpa_separator: str = ";"
    # У stpa-файла первая колонка может быть пустой (Leading ';') — пропускать её
    stpa_skip_first_column: bool = True
    # Имя колонки времени в opers.csv (переименовывается в batch_time при merge)
    opers_join_column: str = "date"


@dataclass
class ValidationConfig:
    """Параметры валидации входных данных (замечание #11)."""

    enabled: bool = True
    duplicate_keys_strategy: str = "fail"
    nan_strategy: str = "fill_median"
    inf_strategy: str = "replace_with_nan"
    max_nan_fraction_per_row: float = 0.9
    max_join_multiplicity: float = 1.5
    strict_binary_target: bool = True
    expected_feature_dtype: str = "float32"


@dataclass
class FeatureEngineeringConfig:
    """Параметры feature engineering (замечание #12)."""

    add_time_features: bool = True
    lag_sizes: list[int] = field(default_factory=lambda: [1, 3, 6])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 12])
    rolling_n_features: int | None = 50
    pca_components: float | int | None = None
    keep_original_features: bool = True


@dataclass
class ExperimentConfig:
    """Параметры сравнения моделей и поиска гиперпараметров (замечание #13)."""

    models_to_compare: list[str] = field(default_factory=lambda: ["mlp"])
    selection_metric: str = "pr_auc"
    cv_folds: int = 0
    optuna_trials: int = 0
    optuna_model: str = "mlp"
    optuna_timeout: int | None = 600


@dataclass
class ThresholdConfig:
    """Оптимизация порога бинарной классификации (замечание #14)."""

    optimize: bool = True
    metric: str = "f1"
    min_precision: float = 0.5
    calibration: str = "none"
    save_curves: bool = True


@dataclass
class MlflowConfig:
    """Трекинг экспериментов через MLflow (замечание #15)."""

    enabled: bool = False
    tracking_uri: str = ""
    experiment_name: str = "gazprom_emergency"
    registered_model_name: str | None = None
    stage: str = "Staging"


@dataclass
class CostMatrixConfig:
    """Стоимости ошибок для cost-sensitive learning (замечание #17)."""

    cost_fn: float = 1000.0
    cost_fp: float = 10.0
    benefit_tp: float = 990.0
    benefit_tn: float = 0.0

    def total_cost(
        self, tp: int, fp: int, fn: int, tn: int
    ) -> float:
        """Вычисляет суммарную бизнес-стоимость матрицы ошибок."""
        return (
            tp * self.benefit_tp
            + tn * self.benefit_tn
            - fn * self.cost_fn
            - fp * self.cost_fp
        )

    @property
    def class_weight(self) -> dict[int, float]:
        """Веса классов для cost-sensitive learning (sklearn-style)."""
        ratio = self.cost_fn / max(self.cost_fp, 1e-9)
        return {0: 1.0, 1: ratio}


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    dropout: float = 0.3
    save_path: str = ""
    architecture: str = "emergency_predictor"


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 50
    batch_size: int = 256
    use_smote: bool = False
    smote_sampling_strategy: float = 0.5
    # Разделение train/val
    test_size: float = 0.2
    split_strategy: str = "chronological"  # "chronological" | "random"
    random_state: int = 42
    seed: int = 42  # глобальный seed для воспроизводимости
    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    # Early stopping
    early_stopping_patience: int = 7
    # Cost-sensitive learning (вес положительного класса в loss)
    pos_weight: float | None = None
    # Режим вычисления pos_weight:
    # "statistical" — из соотношения классов в train (negatives/positives),
    #                 стабильно для обучения (рекомендуется);
    # "cost_matrix" — из cost_matrix (cost_fn / cost_fp), может быть очень
    #                 большим и дестабилизировать обучение;
    # "explicit"    — использовать явно заданное значение pos_weight.
    pos_weight_mode: str = "statistical"
    # Масштабирование признаков: "standard" (z-score) | "minmax" ([0,1]) | "none"
    scaler_type: str = "standard"
    # Предвычислять train-данные в память (с применённым scaler) для скорости.
    # True — быстро, но требует память (529k×3600×4 ≈ 7.6 ГБ);
    # False — ленивая обработка через memmap (медленнее, но меньше память).
    precompute_train: bool = True
    # Gradient clipping (норма градиента; None = выключено)
    gradient_clip_norm: float | None = 1.0
    # LR scheduler: "none" | "reduce_on_plateau" | "cosine"
    scheduler: str = "reduce_on_plateau"
    # Параметры для ReduceLROnPlateau
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6


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
    # Новые секции (замечания #11–#17)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    cost_matrix: CostMatrixConfig = field(default_factory=CostMatrixConfig)


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
        validation=ValidationConfig(**_section_dict(raw, "validation")),
        feature_engineering=FeatureEngineeringConfig(
            **_section_dict(raw, "feature_engineering")
        ),
        experiment=ExperimentConfig(**_section_dict(raw, "experiment")),
        threshold=ThresholdConfig(**_section_dict(raw, "threshold")),
        mlflow=MlflowConfig(**_section_dict(raw, "mlflow")),
        cost_matrix=CostMatrixConfig(**_section_dict(raw, "cost_matrix")),
    )


def load_config(path: str | Path) -> Config:
    """Загружает конфигурацию из YAML-файла."""
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


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Сериализует Config в plain dict (для сохранения рядом с моделью)."""
    import dataclasses

    result: dict[str, Any] = {}
    for f_name in dataclasses.fields(cfg):
        value = getattr(cfg, f_name.name)
        result[f_name.name] = _dataclass_to_dict(value)
    return result


def _dataclass_to_dict(obj: Any) -> Any:
    """Рекурсивно преобразует dataclass в dict/list/scalar."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    return obj