"""Feature engineering (замечание senior review #12).

Интегрирует наработки из legacy-скриптов в основной пайплайн:
- ``filter_fill_nan_stpa.py`` → обработка пропусков (interpolation),
- ``seasonality_analysis.py`` → временные признаки (час, день недели, сезон),
- лаги и скользящие статистики (rolling mean/std/min/max),
- уменьшение размерности через PCA.

Все преобразования реализованы как sklearn-совместимые трансформеры,
которые можно встроить в ``sklearn.Pipeline`` для защиты от data leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


# ============================================================
# Временные признаки (из seasonality_analysis.py)
# ============================================================

# Месяца сезонам (северное полушарие): 0=зима, 1=весна, 2=лето, 3=осень
_MONTH_TO_SEASON = {
    1: 0, 2: 0, 12: 0,  # зима
    3: 1, 4: 1, 5: 1,  # весна
    6: 2, 7: 2, 8: 2,  # лето
    9: 3, 10: 3, 11: 3,  # осень
}


def extract_time_features(batch_times: np.ndarray) -> np.ndarray:
    """Извлекает временные признаки из массива временных меток.

    Признаки (всего 8):
    - hour_sin, hour_cos — час (циклическое кодирование),
    - dayofweek_sin, dayofweek_cos — день недели,
    - month_sin, month_cos — месяц,
    - season_onehot (4 колонки) — one-hot сезона.

    Циклическое кодирование (sin/cos) сохраняет непрерывность:
    23:00 и 00:00 близки во времени.

    Args:
        batch_times: массив ``datetime64[ns]`` временных меток.

    Returns:
        Матрица (n_rows, 8) временных признаков как float32.
    """
    if batch_times.dtype.kind != "M":
        batch_times = batch_times.astype("datetime64[ns]")

    # Извлекаем компоненты через pandas-like операции
    # datetime64 -> unix timestamp -> извлечение через numpy
    years = batch_times.astype("datetime64[Y]")
    months = (batch_times.astype("datetime64[M]").astype(int) % 12) + 1
    # День года и часы
    days_since_epoch = batch_times.astype("datetime64[D]").astype(int)
    hours_raw = (batch_times.astype("datetime64[h]").astype(int))
    hours = hours_raw % 24
    dayofweek = (days_since_epoch + 4) % 7  # 1970-01-01 = четверг = 4

    # Циклическое кодирование
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7.0)
    month_sin = np.sin(2 * np.pi * months / 12.0)
    month_cos = np.cos(2 * np.pi * months / 12.0)

    # Сезон one-hot
    seasons = np.array([_MONTH_TO_SEASON.get(int(m), 0) for m in months])
    season_onehot = np.zeros((len(batch_times), 4), dtype=np.float32)
    for i, s in enumerate(seasons):
        season_onehot[i, int(s)] = 1.0

    features = np.column_stack(
        [
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            month_sin,
            month_cos,
        ]
    ).astype(np.float32)
    features = np.hstack([features, season_onehot])
    return features


def time_feature_names() -> list[str]:
    """Возвращает имена временных признаков (для ``feature_columns``)."""
    return [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "season_winter",
        "season_spring",
        "season_summer",
        "season_autumn",
    ]


# ============================================================
# Лаги и скользящие статистики
# ============================================================


def add_lag_features(
    X: np.ndarray, lag_sizes: list[int], n_features: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Добавляет лаговые признаки для первых ``n_features`` колонок.

    Для каждого признака ``i`` и лага ``L`` создаётся колонка ``f"{feat}_lag{L}"``,
    содержащая значение признака ``L`` строк назад. Первые ``L`` строк заполняются NaN.

    Args:
        X: матрица признаков (n_rows, n_features).
        lag_sizes: список размеров лага.
        n_features: число первых признаков для лагов (None = все).

    Returns:
        Кортеж ``(lag_matrix, feature_names)``.
    """
    n_rows, total_feats = X.shape
    n_feats = min(n_features or total_feats, total_feats)
    lag_list = []
    names: list[str] = []

    for lag in lag_sizes:
        # Сдвигаем вниз на lag строк
        lagged = np.full_like(X[:, :n_feats], np.nan, dtype=np.float32)
        if lag < n_rows:
            lagged[lag:, :] = X[:-lag, :n_feats]
        lag_list.append(lagged)
        names.extend([f"feat{i}_lag{lag}" for i in range(n_feats)])

    if not lag_list:
        return np.empty((n_rows, 0), dtype=np.float32), []

    return np.hstack(lag_list), names


def add_rolling_features(
    X: np.ndarray,
    windows: list[int],
    n_features: int | None = None,
    stats: tuple[str, ...] = ("mean", "std"),
) -> tuple[np.ndarray, list[str]]:
    """Добавляет скользящие статистики для первых ``n_features`` колонок.

    Для каждого окна ``W`` и статистики ``S`` создаётся колонка с rolling-значением.
    Скользящее окно центрировано и исключает текущую строку (предотвращает leakage).

    Args:
        X: матрица признаков (n_rows, n_features).
        windows: список размеров окна.
        n_features: число первых признаков (None = все).
        stats: статистики ("mean", "std", "min", "max").

    Returns:
        Кортеж ``(rolling_matrix, feature_names)``.
    """
    n_rows, total_feats = X.shape
    n_feats = min(n_features or total_feats, total_feats)
    rolling_list = []
    names: list[str] = []

    for window in windows:
        for stat in stats:
            # Используем простое скользящее окно через cumsum для efficiency
            # Исключаем текущую строку (shift на 1), чтобы избежать leakage
            shifted = np.full_like(X[:, :n_feats], np.nan, dtype=np.float32)
            if n_rows > 1:
                shifted[1:, :] = X[:-1, :n_feats]

            if stat == "mean":
                vals = _rolling_stat(shifted, window, np.mean)
            elif stat == "std":
                vals = _rolling_stat(shifted, window, np.std)
            elif stat == "min":
                vals = _rolling_stat(shifted, window, np.nanmin)
            elif stat == "max":
                vals = _rolling_stat(shifted, window, np.nanmax)
            else:
                continue
            rolling_list.append(vals)
            names.extend([f"feat{i}_roll{window}_{stat}" for i in range(n_feats)])

    if not rolling_list:
        return np.empty((n_rows, 0), dtype=np.float32), []

    return np.hstack(rolling_list), names


def _rolling_stat(
    X: np.ndarray, window: int, stat_fn: Any
) -> np.ndarray:
    """Вычисляет rolling-статистику по строкам с NaN-safe логикой."""
    n_rows, n_feats = X.shape
    result = np.full((n_rows, n_feats), np.nan, dtype=np.float32)
    for i in range(window, n_rows):
        chunk = X[i - window : i, :]
        with np.errstate(all="ignore"):
            result[i, :] = stat_fn(chunk, axis=0)
    # Заменяем NaN на 0 для стабильности
    result = np.nan_to_num(result, nan=0.0)
    return result


# ============================================================
# sklearn-совместимый трансформер для полного feature engineering
# ============================================================


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Унифицированный трансформер feature engineering.

    Встраивается в ``sklearn.Pipeline`` и выполняет:
    1. (опционально) добавление временных признаков,
    2. добавление лагов и rolling-статистик,
    3. (опционально) PCA для уменьшения размерности.

    Важно: fit выполняется только на train, transform — на val/test,
    что исключает data leakage.

    Параметры:
        cfg: ``FeatureEngineeringConfig``.
        batch_times: временные метки (нужны только если ``add_time_features``).
    """

    def __init__(
        self,
        cfg: FeatureEngineeringConfig | None = None,
        batch_times: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg or FeatureEngineeringConfig()
        self.batch_times = batch_times
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._feature_names: list[str] = []
        self._n_input_features = 0

    def fit(self, X: np.ndarray, y: Any = None) -> "FeatureEngineer":
        """Обучает scaler и PCA на train-данных."""
        self._n_input_features = X.shape[1]
        X_aug = self._add_features(X, fit_mode=True)

        # Scaler нужен перед PCA
        if self.cfg.pca_components is not None:
            self._scaler = StandardScaler()
            X_aug = self._scaler.fit_transform(X_aug)

            # PCA
            n_components = self.cfg.pca_components
            if isinstance(n_components, float):
                self._pca = PCA(n_components=n_components, random_state=42)
            else:
                self._pca = PCA(n_components=int(n_components), random_state=42)
            X_aug = self._pca.fit_transform(X_aug)
            # Обновляем имена признаков
            self._feature_names = [f"pca_{i}" for i in range(X_aug.shape[1])]
        else:
            self._feature_names = self._compute_feature_names()

        logger.info(
            "FeatureEngineer fitted: %d → %d features",
            self._n_input_features,
            len(self._feature_names),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Преобразует данные (val/test/inference)."""
        X_aug = self._add_features(X, fit_mode=False)
        if self._scaler is not None:
            X_aug = self._scaler.transform(X_aug)
        if self._pca is not None:
            X_aug = self._pca.transform(X_aug)
        return X_aug.astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Объединённый fit + transform (для удобства)."""
        return self.fit(X, y).transform(X)

    @property
    def feature_names(self) -> list[str]:
        """Имена признаков после преобразования."""
        return list(self._feature_names)

    def _add_features(self, X: np.ndarray, fit_mode: bool) -> np.ndarray:
        """Добавляет временные, лаговые и rolling-признаки."""
        parts = [X] if self.cfg.keep_original_features else []
        names = [f"v_{i}" for i in range(X.shape[1])] if self.cfg.keep_original_features else []

        # Временные признаки
        if self.cfg.add_time_features and self.batch_times is not None:
            time_feats = extract_time_features(self.batch_times)
            parts.append(time_feats)
            names.extend(time_feature_names())

        # Лаги
        if self.cfg.lag_sizes:
            lag_X, lag_names = add_lag_features(
                X, self.cfg.lag_sizes, self.cfg.rolling_n_features
            )
            if lag_X.shape[1] > 0:
                # Заполняем NaN нулями (первые строки без истории)
                lag_X = np.nan_to_num(lag_X, nan=0.0)
                parts.append(lag_X)
                names.extend(lag_names)

        # Rolling статистики
        if self.cfg.rolling_windows:
            roll_X, roll_names = add_rolling_features(
                X, self.cfg.rolling_windows, self.cfg.rolling_n_features
            )
            if roll_X.shape[1] > 0:
                parts.append(roll_X)
                names.extend(roll_names)

        if fit_mode and not self.cfg.pca_components:
            self._feature_names = names

        if not parts:
            return X
        return np.hstack(parts)

    def _compute_feature_names(self) -> list[str]:
        """Вычисляет имена признаков без PCA."""
        names: list[str] = []
        if self.cfg.keep_original_features:
            names.extend([f"v_{i}" for i in range(self._n_input_features)])
        if self.cfg.add_time_features:
            names.extend(time_feature_names())
        if self.cfg.lag_sizes:
            n_feats = min(
                self.cfg.rolling_n_features or self._n_input_features,
                self._n_input_features,
            )
            for lag in self.cfg.lag_sizes:
                names.extend([f"feat{i}_lag{lag}" for i in range(n_feats)])
        if self.cfg.rolling_windows:
            n_feats = min(
                self.cfg.rolling_n_features or self._n_input_features,
                self._n_input_features,
            )
            for w in self.cfg.rolling_windows:
                names.extend([f"feat{i}_roll{w}_mean" for i in range(n_feats)])
                names.extend([f"feat{i}_roll{w}_std" for i in range(n_feats)])
        return names


# ============================================================
# Обработка пропусков (интеграция filter_fill_nan_stpa.py)
# ============================================================


def interpolate_nan_rows(
    X: np.ndarray, feature_mask: np.ndarray | None = None
) -> np.ndarray:
    """Заполняет строки с NaN интерполяцией соседних строк.

    Реплика логики из ``legacy/filter_fill_nan_stpa.py``:
    - если вся строка NaN → заменяем средним соседних (prev + next) / 2,
    - первая строка → следующая,
    - последняя строка → предыдущая.

    Это сохраняет временной контекст и лучше, чем fillna(mean), для временных рядов.

    Args:
        X: матрица признаков (n_rows, n_features).
        feature_mask: маска колонок для проверки (None = все).

    Returns:
        Очищенная матрица.
    """
    X = X.copy()
    n_rows = X.shape[0]
    check_cols = feature_mask if feature_mask is not None else np.arange(X.shape[1])

    for i in range(n_rows):
        row = X[i, check_cols]
        if np.all(np.isnan(row)):
            if i == 0 and n_rows > 1:
                X[i, :] = X[i + 1, :]
            elif i == n_rows - 1 and n_rows > 1:
                X[i, :] = X[i - 1, :]
            elif n_rows > 2:
                X[i, :] = (X[i - 1, :] + X[i + 1, :]) / 2.0
            else:
                X[i, :] = 0.0
    return X