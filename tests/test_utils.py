"""Тесты для utils.py: seed, JSON scaler, инкрементальный fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from projects.gazprom_emergency.utils import (
    fit_minmax_incremental,
    fit_scaler,
    fit_standard_incremental,
    load_scaler_json,
    save_scaler_json,
    set_seed,
)


# ============================================================
# set_seed — воспроизводимость
# ============================================================

class TestSetSeed:
    """Тестирует фиксацию seed'ов для воспроизводимости."""

    def test_seed_sets_numpy(self):
        """После set_seed numpy выдаёт одинаковые случайные числа."""
        set_seed(42)
        a = np.random.rand(10)

        set_seed(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)

    def test_seed_sets_torch(self):
        """После set_seed torch выдаёт одинаковые случайные числа."""
        import torch

        set_seed(42)
        a = torch.rand(10)

        set_seed(42)
        b = torch.rand(10)

        torch.testing.assert_close(a, b)

    def test_different_seeds_produce_different_results(self):
        """Разные seed'ы дают разные последовательности."""
        set_seed(42)
        a = np.random.rand(10)

        set_seed(123)
        b = np.random.rand(10)

        assert not np.allclose(a, b)

    def test_cudnn_deterministic_flag(self):
        """После set_seed cudnn.deterministic = True."""
        import torch

        set_seed(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


# ============================================================
# JSON scaler: save/load (MinMaxScaler)
# ============================================================

class TestScalerJSON:
    """Тестирует безопасную (де)сериализацию MinMaxScaler через JSON."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Save → load восстанавливает идентичный scaler."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(X)

        path = tmp_path / "test.scaler.json"
        save_scaler_json(scaler, path)
        loaded = load_scaler_json(path)

        # Параметры должны совпадать
        np.testing.assert_array_almost_equal(scaler.min_, loaded.min_)
        np.testing.assert_array_almost_equal(scaler.scale_, loaded.scale_)
        np.testing.assert_array_almost_equal(scaler.data_min_, loaded.data_min_)
        np.testing.assert_array_almost_equal(scaler.data_max_, loaded.data_max_)

    def test_transform_produces_same_result(self, tmp_path: Path):
        """Загруженный scaler даёт тот же transform, что оригинал."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(X)

        path = tmp_path / "test.scaler.json"
        save_scaler_json(scaler, path)
        loaded = load_scaler_json(path)

        X_test = np.array([[2.5, 25], [3.5, 35]], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            scaler.transform(X_test), loaded.transform(X_test)
        )

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Загрузка несуществующего файла вызывает FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Scaler file not found"):
            load_scaler_json(tmp_path / "nonexistent.json")

    def test_scaler_creates_parent_dir(self, tmp_path: Path):
        """save_scaler_json создаёт родительские директории."""
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(X)

        path = tmp_path / "deep" / "nested" / "dir" / "scaler.json"
        save_scaler_json(scaler, path)
        assert path.exists()


# ============================================================
# JSON scaler: save/load (StandardScaler)
# ============================================================

class TestStandardScalerJSON:
    """Тестирует сериализацию StandardScaler через JSON."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Save → load восстанавливает идентичный StandardScaler."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(X)

        path = tmp_path / "test.scaler.json"
        save_scaler_json(scaler, path)
        loaded = load_scaler_json(path)

        assert isinstance(loaded, StandardScaler)
        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, loaded.scale_)

    def test_transform_produces_same_result(self, tmp_path: Path):
        """Загруженный StandardScaler даёт тот же transform."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(X)

        path = tmp_path / "test.scaler.json"
        save_scaler_json(scaler, path)
        loaded = load_scaler_json(path)

        X_test = np.array([[2.5, 25], [3.5, 35]], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            scaler.transform(X_test), loaded.transform(X_test)
        )

    def test_unsupported_scaler_raises(self, tmp_path: Path):
        """Неподдерживаемый тип scaler вызывает TypeError."""
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler().fit(np.array([[1, 2], [3, 4]]))
        with pytest.raises(TypeError, match="Unsupported scaler type"):
            save_scaler_json(scaler, tmp_path / "test.scaler.json")


# ============================================================
# fit_minmax_incremental — чанковый fit
# ============================================================

class TestIncrementalFit:
    """Тестирует инкрементальный fit MinMaxScaler."""

    def test_incremental_matches_standard(self, tmp_path: Path):
        """Инкрементальный fit даёт те же результаты, что стандартный."""
        n_rows, n_features = 1000, 10
        data = np.random.rand(n_rows, n_features).astype(np.float32) * 100

        # Создаём memmap
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
        X_mm[:] = data
        X_mm.flush()

        # Перечитываем
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))

        # Стандартный fit
        standard_scaler = MinMaxScaler()
        standard_scaler.fit(data)

        # Инкрементальный fit
        indices = np.arange(n_rows)
        incremental_scaler = fit_minmax_incremental(X_mm, indices, chunk_size=100)

        np.testing.assert_array_almost_equal(
            standard_scaler.data_min_, incremental_scaler.data_min_, decimal=5
        )
        np.testing.assert_array_almost_equal(
            standard_scaler.data_max_, incremental_scaler.data_max_, decimal=5
        )

    def test_incremental_on_subset(self, tmp_path: Path):
        """Инкрементальный fit на подмножестве индексов считает min/max только по ним."""
        data = np.array(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [100, 1000]],
            dtype=np.float32,
        )
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        # Только первые 5 строк (без строки [100, 1000])
        train_idx = np.arange(5)
        scaler = fit_minmax_incremental(X_mm, train_idx, chunk_size=2)

        assert scaler.data_min_[0] == 1
        assert scaler.data_max_[0] == 5
        assert scaler.data_min_[1] == 10
        assert scaler.data_max_[1] == 50

    def test_incremental_handles_constant_column(self, tmp_path: Path):
        """Постоянная колонка не вызывает деления на ноль."""
        data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        scaler = fit_minmax_incremental(X_mm, np.arange(3))
        # scale для константной колонки не должен быть inf/nan
        assert np.isfinite(scaler.scale_).all()


# ============================================================
# fit_standard_incremental — чанковый fit StandardScaler
# ============================================================

class TestStandardIncrementalFit:
    """Тестирует инкрементальный fit StandardScaler."""

    def test_incremental_matches_standard(self, tmp_path: Path):
        """Инкрементальный StandardScaler даёт те же mean/std, что обычный."""
        n_rows, n_features = 1000, 10
        rng = np.random.RandomState(42)
        data = rng.rand(n_rows, n_features).astype(np.float32) * 100

        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))

        # Эталонный StandardScaler
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)

        # Инкрементальный fit
        indices = np.arange(n_rows)
        incremental_scaler = fit_standard_incremental(X_mm, indices, chunk_size=100)

        np.testing.assert_array_almost_equal(
            standard_scaler.mean_, incremental_scaler.mean_, decimal=4
        )
        np.testing.assert_array_almost_equal(
            standard_scaler.scale_, incremental_scaler.scale_, decimal=4
        )

    def test_incremental_on_subset(self, tmp_path: Path):
        """fit на подмножестве индексов считает статистику только по ним."""
        data = np.array(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [1000, 10000]],
            dtype=np.float32,
        )
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        # Только первые 5 строк
        train_idx = np.arange(5)
        scaler = fit_standard_incremental(X_mm, train_idx, chunk_size=2)

        standard_scaler = StandardScaler()
        standard_scaler.fit(data[:5])

        np.testing.assert_array_almost_equal(
            scaler.mean_, standard_scaler.mean_, decimal=4
        )

    def test_constant_column_no_division_by_zero(self, tmp_path: Path):
        """Постоянная колонка не вызывает деления на ноль."""
        data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        scaler = fit_standard_incremental(X_mm, np.arange(3))
        assert np.isfinite(scaler.scale_).all()
        # Для константной колонки scale_ должен быть 1.0 (1/1)
        assert scaler.scale_[0] == 1.0


# ============================================================
# fit_scaler — универсальная фабрика
# ============================================================

class TestFitScalerFactory:
    """Тестирует универсальную фабрику fit_scaler."""

    def test_fit_scaler_standard(self, tmp_path: Path):
        """fit_scaler(type='standard') возвращает StandardScaler."""
        data = np.random.rand(100, 5).astype(np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        scaler = fit_scaler(X_mm, np.arange(100), scaler_type="standard")
        assert isinstance(scaler, StandardScaler)

    def test_fit_scaler_minmax(self, tmp_path: Path):
        """fit_scaler(type='minmax') возвращает MinMaxScaler."""
        data = np.random.rand(100, 5).astype(np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        scaler = fit_scaler(X_mm, np.arange(100), scaler_type="minmax")
        assert isinstance(scaler, MinMaxScaler)

    def test_fit_scaler_none(self, tmp_path: Path):
        """fit_scaler(type='none') возвращает None (без масштабирования)."""
        data = np.random.rand(100, 5).astype(np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        scaler = fit_scaler(X_mm, np.arange(100), scaler_type="none")
        assert scaler is None

    def test_fit_scaler_unknown_raises(self, tmp_path: Path):
        """Неизвестный scaler_type вызывает ValueError."""
        data = np.random.rand(100, 5).astype(np.float32)
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=data.shape)
        X_mm[:] = data
        X_mm.flush()
        X_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=data.shape)

        with pytest.raises(ValueError, match="Unknown scaler_type"):
            fit_scaler(X_mm, np.arange(100), scaler_type="unknown")