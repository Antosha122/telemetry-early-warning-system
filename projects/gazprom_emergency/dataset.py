"""Dataset и разбиение данных поверх np.memmap (без загрузки в ОЗУ)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .data import COL_BATCH_TIME, load_batch_times

logger = logging.getLogger(__name__)


# ============================================================
# Хронологическое разбиение по batch_time (без data leakage)
# ============================================================

def chronological_split_indices(
    batch_times: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Разбивает индексы по времени: train — первые (1-test_size), val — последние test_size.

    Защита от data leakage: модель никогда не видит «будущие» записи при обучении.

    Args:
        batch_times: массив временных меток для каждой строки.
        test_size: доля валидационной выборки (по времени).

    Returns:
        (train_indices, val_indices) — отсортированные массивы индексов.
    """
    n = len(batch_times)
    # Сортируем по времени, получаем порядок строк
    sorted_order = np.argsort(batch_times, kind="stable")

    split_point = int(n * (1.0 - test_size))
    train_idx = np.sort(sorted_order[:split_point])
    val_idx = np.sort(sorted_order[split_point:])

    logger.info(
        "Chronological split: train=%d rows (up to %s), val=%d rows (from %s)",
        len(train_idx),
        batch_times[sorted_order[split_point - 1]] if split_point > 0 else "N/A",
        len(val_idx),
        batch_times[sorted_order[split_point]] if split_point < n else "N/A",
    )
    return train_idx, val_idx


def train_val_split(
    X: np.memmap,
    y: np.memmap,
    batch_times_path: str | Path,
    test_size: float = 0.2,
    *,
    strategy: str = "chronological",
) -> tuple[np.ndarray, np.ndarray]:
    """Разбивает данные на train/val по выбранной стратегии.

    Args:
        X: memmap признаков (нужен только для размера).
        y: memmap целевой (нужен только для размера).
        batch_times_path: путь к файлу с временными метками.
        test_size: доля val.
        strategy: "chronological" (по умолчанию) — разбиение по времени.

    Returns:
        (train_indices, val_indices).
    """
    n_rows = X.shape[0]
    assert y.shape[0] == n_rows, f"X rows ({n_rows}) != y rows ({y.shape[0]})"

    if strategy == "chronological":
        batch_times = load_batch_times(batch_times_path)
        assert len(batch_times) == n_rows, (
            f"batch_times length ({len(batch_times)}) != data rows ({n_rows})"
        )
        return chronological_split_indices(batch_times, test_size)

    raise ValueError(f"Unknown split strategy: {strategy!r}")


# ============================================================
# ScalerMixin: ленивое масштабирование в __getitem__
# ============================================================

class _ScalerTransform:
    """Обёртка, применяющая transform к numpy-чанку."""

    def __init__(self, scaler) -> None:  # type: ignore[no-untyped-def]
        self.scaler = scaler

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


# ============================================================
# MemmapDataset: PyTorch Dataset поверх np.memmap
# ============================================================

class MemmapDataset(Dataset):
    """PyTorch Dataset поверх np.memmap — не загружает данные целиком в ОЗУ.

    Чтение происходит постранично через __getitem__; scaler.transform применяется
    лениво к каждому батчу, а не ко всей матрице сразу.
    """

    def __init__(
        self,
        X: np.memmap,
        y: np.memmap,
        indices: np.ndarray,
        scaler=None,  # type: ignore[no-untyped-def]
    ) -> None:
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        row_idx = int(self.indices[idx])
        x_row = np.asarray(self.X[row_idx], dtype=np.float32)
        if self.scaler is not None:
            # transform expects 2D: reshape (1, n_features) -> back to 1D
            x_row = self.scaler.transform(x_row.reshape(1, -1)).ravel()
        y_val = np.float32(self.y[row_idx])
        return x_row, y_val