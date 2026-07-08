"""Тесты для dataset.py: хронологическое разбиение, MemmapDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.gazprom_emergency.dataset import (
    MemmapDataset,
    chronological_split_indices,
)


# ============================================================
# Хронологическое разбиение (без data leakage)
# ============================================================

class TestChronologicalSplit:
    """Тестирует разбиение данных по времени."""

    def test_train_before_val_chronologically(self):
        """Все train-записи раньше val по времени (нет заглядывания в будущее)."""
        batch_times = np.array(
            [
                "2023-01-05",
                "2023-01-01",
                "2023-01-10",
                "2023-01-03",
                "2023-01-08",
                "2023-01-12",
            ],
            dtype="datetime64[D]",
        )

        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.5)

        train_times = batch_times[train_idx]
        val_times = batch_times[val_idx]

        # Все train-времена должны быть <= всем val-временам
        assert train_times.max() <= val_times.min(), (
            f"Train max ({train_times.max()}) > val min ({val_times.min()}) — data leakage!"
        )

    def test_split_proportions(self):
        """Размер train/val соответствует test_size."""
        batch_times = np.arange("2023-01-01", "2023-04-11", dtype="datetime64[D]")

        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.2)
        assert len(train_idx) == 80
        assert len(val_idx) == 20

    def test_split_80_20(self):
        """Стандартное разбиение 80/20."""
        batch_times = np.arange("2023-01-01", "2023-01-11", dtype="datetime64[D]")

        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.2)
        assert len(train_idx) == 8
        assert len(val_idx) == 2

    def test_indices_are_disjoint(self):
        """Train и val индексы не пересекаются."""
        batch_times = np.arange("2023-01-01", "2023-02-20", dtype="datetime64[D]")
        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.3)

        intersection = np.intersect1d(train_idx, val_idx)
        assert len(intersection) == 0, f"Overlap: {intersection}"

    def test_all_indices_covered(self):
        """Все индексы распределены между train и val."""
        n = 100
        batch_times = np.arange("2023-01-01", "2023-04-11", dtype="datetime64[D]")
        assert len(batch_times) == n
        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.25)

        all_idx = np.union1d(train_idx, val_idx)
        assert len(all_idx) == n

    def test_random_order_input_handled(self):
        """Перемешанный вход сортируется корректно."""
        batch_times = np.array(
            ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-05", "2023-01-04"],
            dtype="datetime64[D]",
        )
        train_idx, val_idx = chronological_split_indices(batch_times, test_size=0.4)

        # Первые 3 (60%) — самые ранние по времени
        train_times = batch_times[train_idx]
        val_times = batch_times[val_idx]
        assert train_times.max() <= val_times.min()


# ============================================================
# MemmapDataset
# ============================================================

class TestMemmapDataset:
    """Тестирует PyTorch Dataset поверх np.memmap."""

    def test_dataset_length(self, tmp_path: Path):
        """__len__ возвращает число индексов."""
        n_rows, n_features = 10, 5
        X = np.random.rand(n_rows, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_rows).astype(np.float32)

        x_path = tmp_path / "X.npy"
        y_path = tmp_path / "y.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=X.shape)
        y_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=y.shape)
        X_mm[:] = X
        y_mm[:] = y
        X_mm.flush()
        y_mm.flush()

        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=X.shape)
        y_mm = np.memmap(y_path, dtype=np.float32, mode="r", shape=y.shape)

        indices = np.array([0, 2, 4, 6])
        ds = MemmapDataset(X_mm, y_mm, indices)
        assert len(ds) == 4

    def test_getitem_returns_correct_row(self, tmp_path: Path):
        """__getitem__ возвращает правильную строку по индексу."""
        n_rows, n_features = 5, 3
        X = np.arange(n_rows * n_features).reshape(n_rows, n_features).astype(np.float32)
        y = np.arange(n_rows).astype(np.float32)

        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=X.shape)
        X_mm[:] = X
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=X.shape)

        y_path = tmp_path / "y.npy"
        y_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=y.shape)
        y_mm[:] = y
        y_mm.flush()
        y_mm = np.memmap(y_path, dtype=np.float32, mode="r", shape=y.shape)

        indices = np.array([0, 2, 4])
        ds = MemmapDataset(X_mm, y_mm, indices)

        x_val, y_val = ds[1]  # indices[1] = 2
        np.testing.assert_array_equal(x_val, X[2])
        assert y_val == y[2]

    def test_getitem_with_scaler(self, tmp_path: Path):
        """С scaler __getitem__ применяет transform к строке."""
        from sklearn.preprocessing import MinMaxScaler

        n_rows, n_features = 4, 2
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)

        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=X.shape)
        X_mm[:] = X
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=X.shape)

        y_mm = np.memmap(tmp_path / "y.npy", dtype=np.float32, mode="w+", shape=(n_rows,))
        y_mm[:] = np.zeros(n_rows, dtype=np.float32)
        y_mm.flush()
        y_mm = np.memmap(tmp_path / "y.npy", dtype=np.float32, mode="r", shape=(n_rows,))

        scaler = MinMaxScaler()
        scaler.fit(X)

        indices = np.array([0, 1])
        ds = MemmapDataset(X_mm, y_mm, indices, scaler=scaler)

        x_val, _ = ds[0]
        # MinMax scaled: (val - min) / (max - min)
        np.testing.assert_array_almost_equal(x_val, scaler.transform(X[0:1]).ravel())