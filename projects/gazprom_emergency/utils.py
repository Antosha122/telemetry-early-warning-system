"""Общие утилиты: фиксация seed, безопасная сериализация scaler."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

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
# Безопасная сериализация MinMaxScaler через JSON
# ============================================================

def save_scaler_json(scaler: MinMaxScaler, path: str | Path) -> None:
    """Сохраняет параметры MinMaxScaler в JSON (без pickle / RCE-рисков).

    Args:
        scaler: обученный MinMaxScaler.
        path: путь к файлу (.scaler.json).
    """
    path = Path(path)
    data: dict[str, Any] = {
        "feature_range": list(scaler.feature_range),
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_) if scaler.n_samples_seen_ is not None else 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Scaler saved to %s (JSON)", path)


def load_scaler_json(path: str | Path) -> MinMaxScaler:
    """Загружает MinMaxScaler из JSON.

    Args:
        path: путь к файлу (.scaler.json).

    Returns:
        Восстановленный MinMaxScaler с заполненными атрибутами.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature_range = tuple(data["feature_range"])
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


# ============================================================
# Инкрементальный fit MinMaxScaler (без загрузки всех данных в ОЗУ)
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