"""Общие утилиты: фиксация seed, безопасная сериализация scaler."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


# ============================================================
# Воспроизводимость
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Фиксирует все источники случайности для воспроизводимости.

    Args:
        seed: целочисленный seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Детерминированные алгоритмы cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (cudnn.deterministic=True)", seed)


# ============================================================
# Безопасная сериализация scaler через JSON (StandardScaler + MinMaxScaler)
# ============================================================

def save_scaler_json(scaler: Any, path: str | Path) -> None:
    """Сохраняет параметры scaler в JSON (без pickle / RCE-рисков).

    Поддерживает ``MinMaxScaler`` и ``StandardScaler``. Тип определяется
    по наличию атрибутов и сохраняется в поле ``scaler_type``.

    Args:
        scaler: обученный ``MinMaxScaler`` или ``StandardScaler``.
        path: путь к файлу (.scaler.json).
    """
    path = Path(path)
    data: dict[str, Any] = {
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_)
        if getattr(scaler, "n_samples_seen_", None) is not None
        else 0,
    }

    if isinstance(scaler, StandardScaler):
        data["scaler_type"] = "standard"
        data["mean_"] = np.asarray(scaler.mean_).tolist()
        # scale_ = std; transform = (x - mean_) / scale_
        data["scale_"] = np.asarray(scaler.scale_).tolist()
        data["var_"] = np.asarray(scaler.var_).tolist()
    elif isinstance(scaler, MinMaxScaler):
        data["scaler_type"] = "minmax"
        data["feature_range"] = list(scaler.feature_range)
        data["min_"] = scaler.min_.tolist()
        data["scale_"] = scaler.scale_.tolist()
        data["data_min_"] = scaler.data_min_.tolist()
        data["data_max_"] = scaler.data_max_.tolist()
        data["data_range_"] = scaler.data_range_.tolist()
    else:
        raise TypeError(
            f"Unsupported scaler type: {type(scaler).__name__}. "
            "Supported: MinMaxScaler, StandardScaler."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Scaler (%s) saved to %s (JSON)", data["scaler_type"], path)


def load_scaler_json(path: str | Path) -> Any:
    """Загружает scaler из JSON.

    Автоматически определяет тип (``standard`` или ``minmax``) по полю
    ``scaler_type``. Для обратной совместимости, если поле отсутствует,
    считается, что это ``MinMaxScaler``.

    Args:
        path: путь к файлу (.scaler.json).

    Returns:
        Восстановленный ``MinMaxScaler`` или ``StandardScaler``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scaler_type = data.get("scaler_type", "minmax")

    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(data["mean_"], dtype=np.float64)
        scaler.scale_ = np.asarray(data["scale_"], dtype=np.float64)
        scaler.var_ = np.asarray(data.get("var_", data["scale_"]) , dtype=np.float64)
        scaler.n_features_in_ = int(data["n_features_in_"])
        scaler.n_samples_seen_ = int(data.get("n_samples_seen_", 0))
        return scaler

    if scaler_type == "minmax":
        feature_range = tuple(data.get("feature_range", (0.0, 1.0)))
        scaler = MinMaxScaler(feature_range=feature_range)
        # Подставляем вычисленные параметры, минуя fit()
        scaler.min_ = np.asarray(data["min_"], dtype=np.float64)
        scaler.scale_ = np.asarray(data["scale_"], dtype=np.float64)
        scaler.data_min_ = np.asarray(data["data_min_"], dtype=np.float64)
        scaler.data_max_ = np.asarray(data["data_max_"], dtype=np.float64)
        scaler.data_range_ = np.asarray(data["data_range_"], dtype=np.float64)
        scaler.n_features_in_ = int(data["n_features_in_"])
        scaler.n_samples_seen_ = int(data.get("n_samples_seen_", 0))
        return scaler

    raise ValueError(f"Unknown scaler_type in JSON: {scaler_type!r}")


# ============================================================
# Инкрементальный fit scaler (без загрузки всех данных в ОЗУ)
# ============================================================

def fit_minmax_incremental(
    X: np.memmap,
    indices: np.ndarray,
    chunk_size: int = 50_000,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> MinMaxScaler:
    """Вычисляет MinMaxScaler чанками по индексам без загрузки всех данных в ОЗУ.

    Делает один проход по train-индексам, агрегируя min и max по каждому признаку.

    Args:
        X: memmap-матрица признаков (n_rows, n_features).
        indices: индексы строк обучающей выборки.
        chunk_size: размер чанка для чтения.
        feature_range: целевой диапазон масштабирования.

    Returns:
        Обученный MinMaxScaler с заполненными атрибутами.
    """
    n_features = X.shape[1]
    global_min = np.full(n_features, np.inf, dtype=np.float64)
    global_max = np.full(n_features, -np.inf, dtype=np.float64)
    total_seen = 0

    sorted_indices = np.sort(indices)
    for start in range(0, len(sorted_indices), chunk_size):
        chunk_idx = sorted_indices[start : start + chunk_size]
        chunk = X[chunk_idx]
        global_min = np.minimum(global_min, np.nanmin(chunk, axis=0))
        global_max = np.maximum(global_max, np.nanmax(chunk, axis=0))
        total_seen += len(chunk_idx)

    # Заменяем inf на 0 (если колонка константна или пуста)
    data_range = global_max - global_min
    data_range[data_range == 0] = 1.0

    lo, hi = feature_range
    scale = (hi - lo) / data_range
    min_ = lo - global_min * scale

    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.min_ = min_.astype(np.float64)
    scaler.scale_ = scale.astype(np.float64)
    scaler.data_min_ = global_min
    scaler.data_max_ = global_max
    scaler.data_range_ = data_range
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = total_seen

    logger.info(
        "MinMaxScaler fitted incrementally on %d rows (chunk_size=%d)",
        total_seen,
        chunk_size,
    )
    return scaler


def fit_standard_incremental(
    X: np.memmap,
    indices: np.ndarray,
    chunk_size: int = 50_000,
) -> StandardScaler:
    """Вычисляет StandardScaler чанками по индексам без загрузки всех данных в ОЗУ.

    Использует численно устойчивый онлайн-алгоритм (Welford) в пакетном режиме:
    агрегирует sum и sum_of_squares по каждому признаку за один проход по
    train-индексам. Это эквивалентно ``StandardScaler().fit(X_train)`` по
    результату, но требует память только для одного чанка.

    Args:
        X: memmap-матрица признаков (n_rows, n_features).
        indices: индексы строк обучающей выборки.
        chunk_size: размер чанка для чтения.

    Returns:
        Обученный StandardScaler с заполненными атрибутами.
    """
    n_features = X.shape[1]
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)
    total_seen = 0

    sorted_indices = np.sort(indices)
    for start in range(0, len(sorted_indices), chunk_size):
        chunk_idx = sorted_indices[start : start + chunk_size]
        chunk = np.asarray(X[chunk_idx], dtype=np.float64)
        # NaN-safe агрегация
        total_sum += np.nansum(chunk, axis=0)
        total_sq += np.nansum(chunk * chunk, axis=0)
        total_seen += len(chunk_idx)

    if total_seen == 0:
        raise ValueError("Cannot fit StandardScaler: empty train indices")

    mean = total_sum / total_seen
    # variance = E[x^2] - (E[x])^2
    variance = (total_sq / total_seen) - (mean * mean)
    # Защита от отрицательных значений из-за численной неустойчивости
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    # Защита от деления на ноль для константных признаков
    std[std == 0.0] = 1.0

    scaler = StandardScaler()
    scaler.mean_ = mean.astype(np.float64)
    # В sklearn StandardScaler: scale_ = std, transform = (x - mean_) / scale_
    scaler.scale_ = std.astype(np.float64)
    scaler.var_ = variance.astype(np.float64)
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = total_seen

    logger.info(
        "StandardScaler fitted incrementally on %d rows (chunk_size=%d)",
        total_seen,
        chunk_size,
    )
    return scaler


def fit_scaler(
    X: np.memmap,
    indices: np.ndarray,
    scaler_type: str = "standard",
    chunk_size: int = 50_000,
) -> Any:
    """Универсальная фабрика инкрементального fit'а scaler'а.

    Args:
        X: memmap-матрица признаков (n_rows, n_features).
        indices: индексы строк обучающей выборки.
        scaler_type: "standard" | "minmax".
        chunk_size: размер чанка для чтения.

    Returns:
        Обученный ``StandardScaler`` или ``MinMaxScaler``.
    """
    if scaler_type == "standard":
        return fit_standard_incremental(X, indices, chunk_size=chunk_size)
    if scaler_type == "minmax":
        return fit_minmax_incremental(X, indices, chunk_size=chunk_size)
    if scaler_type == "none":
        return None
    raise ValueError(
        f"Unknown scaler_type: {scaler_type!r}. Supported: 'standard', 'minmax', 'none'."
    )