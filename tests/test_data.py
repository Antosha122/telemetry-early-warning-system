"""Тесты для data.py: merge, memmap, batch_time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from projects.gazprom_emergency.config import DataConfig
from projects.gazprom_emergency.data import (
    COL_BATCH_TIME,
    COL_IS_EMERGENCY,
    _feature_columns,
    get_processed_paths,
    load_batch_times,
    load_memmap,
    merge_to_memmap,
)


# ============================================================
# _feature_columns
# ============================================================

class TestFeatureColumns:
    """Тестирует извлечение списка признаков v_*."""

    def test_returns_sorted_features(self):
        """Колонки v_* сортируются по номеру."""
        schema = {
            "batch_time": str,
            "v_2": float,
            "v_0": float,
            "v_1": float,
            "is_emergency": int,
        }
        cols = _feature_columns(schema)
        assert cols == ["v_0", "v_1", "v_2"]

    def test_ignores_non_v_columns(self):
        """Не-v_ колонки не попадают в список."""
        schema = {"batch_time": str, "other_col": float, "v_0": float, "v_1": float}
        cols = _feature_columns(schema)
        assert cols == ["v_0", "v_1"]

    def test_empty_schema(self):
        """Пустая схема возвращает пустой список."""
        cols = _feature_columns({})
        assert cols == []


# ============================================================
# merge_to_memmap
# ============================================================

class TestMergeToMemmap:
    """Тестирует объединение opers + stpa в memmap."""

    def test_merge_creates_three_files(
        self, tmp_path: Path, sample_opers_df: pl.DataFrame, sample_stpa_df: pl.DataFrame
    ):
        """merge_to_memmap создаёт X_merged.npy, y_merged.npy, t_merged.npy."""
        source_dir = tmp_path / "source"
        processed_dir = tmp_path / "processed"
        source_dir.mkdir()
        processed_dir.mkdir()

        sample_opers_df.write_csv(source_dir / "opers.csv")
        sample_stpa_df.write_csv(source_dir / "stpa.csv")

        cfg = DataConfig(
            source_dir=str(source_dir),
            processed_dir=str(processed_dir),
            chunk_size=10,
        )
        x_path, y_path, n_features = merge_to_memmap(cfg)

        assert x_path.exists()
        assert y_path.exists()
        assert (processed_dir / "t_merged.npy").exists()
        assert n_features == 5

    def test_merge_data_correctness(
        self, tmp_path: Path, sample_opers_df: pl.DataFrame, sample_stpa_df: pl.DataFrame
    ):
        """Данные в memmap соответствуют исходным (features + target)."""
        source_dir = tmp_path / "source"
        processed_dir = tmp_path / "processed"
        source_dir.mkdir()
        processed_dir.mkdir()

        sample_opers_df.write_csv(source_dir / "opers.csv")
        sample_stpa_df.write_csv(source_dir / "stpa.csv")

        cfg = DataConfig(
            source_dir=str(source_dir),
            processed_dir=str(processed_dir),
            chunk_size=10,
        )
        x_path, y_path, n_features = merge_to_memmap(cfg)

        X, y = load_memmap(x_path, y_path, n_features)

        assert X.shape == (5, 5)
        assert y.shape == (5,)

        # y должен содержать is_emergency из opers
        assert set(y.tolist()) == {0.0, 1.0}
        # Два аварийных случая (из sample_opers_df)
        assert (y == 1.0).sum() == 2

    def test_merge_preserves_batch_time(
        self, tmp_path: Path, sample_opers_df: pl.DataFrame, sample_stpa_df: pl.DataFrame
    ):
        """batch_time сохраняется в t_merged.npy."""
        source_dir = tmp_path / "source"
        processed_dir = tmp_path / "processed"
        source_dir.mkdir()
        processed_dir.mkdir()

        sample_opers_df.write_csv(source_dir / "opers.csv")
        sample_stpa_df.write_csv(source_dir / "stpa.csv")

        cfg = DataConfig(
            source_dir=str(source_dir),
            processed_dir=str(processed_dir),
            chunk_size=10,
        )
        merge_to_memmap(cfg)

        t_path = processed_dir / "t_merged.npy"
        batch_times = load_batch_times(t_path)

        assert len(batch_times) == 5
        # Все времена из января 2023
        assert (batch_times >= np.datetime64("2023-01-01")).all()
        assert (batch_times <= np.datetime64("2023-01-02")).all()

    def test_merge_file_not_found_opers(self, tmp_path: Path):
        """Отсутствует opers.csv → FileNotFoundError."""
        source_dir = tmp_path / "source"
        processed_dir = tmp_path / "processed"
        source_dir.mkdir()

        # Создаём только stpa.csv
        pl.DataFrame({"batch_time": ["2023-01-01"], "v_0": [1.0]}).write_csv(
            source_dir / "stpa.csv"
        )

        cfg = DataConfig(
            source_dir=str(source_dir),
            processed_dir=str(processed_dir),
        )
        with pytest.raises(FileNotFoundError, match="opers file not found"):
            merge_to_memmap(cfg)

    def test_merge_file_not_found_stpa(self, tmp_path: Path):
        """Отсутствует stpa.csv → FileNotFoundError."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        pl.DataFrame({"batch_time": ["2023-01-01"], "is_emergency": [0]}).write_csv(
            source_dir / "opers.csv"
        )

        cfg = DataConfig(
            source_dir=str(source_dir),
            processed_dir=str(tmp_path / "processed"),
        )
        with pytest.raises(FileNotFoundError, match="stpa file not found"):
            merge_to_memmap(cfg)


# ============================================================
# load_memmap / load_batch_times
# ============================================================

class TestLoadFunctions:
    """Тестирует функции загрузки memmap."""

    def test_load_memmap_shape(self, tmp_path: Path):
        """load_memmap возвращает корректную форму."""
        n_rows, n_features = 10, 3
        x_path = tmp_path / "X.npy"
        X_mm = np.memmap(
            x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features)
        )
        X_mm.flush()

        y_path = tmp_path / "y.npy"
        y_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(n_rows,))
        y_mm.flush()

        X, y = load_memmap(x_path, y_path, n_features)
        assert X.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)

    def test_load_batch_times_nonexistent(self, tmp_path: Path):
        """load_batch_times вызывает FileNotFoundError для несуществующего файла."""
        with pytest.raises(FileNotFoundError, match="batch_time memmap not found"):
            load_batch_times(tmp_path / "nonexistent.npy")


# ============================================================
# get_processed_paths
# ============================================================

class TestGetProcessedPaths:
    """Тестирует get_processed_paths."""

    def test_returns_three_paths(self, tmp_path: Path):
        """Возвращает пути к X, y и t (batch_time)."""
        cfg = DataConfig(processed_dir=str(tmp_path))
        x_path, y_path, t_path = get_processed_paths(cfg)

        assert x_path.name == "X_merged.npy"
        assert y_path.name == "y_merged.npy"
        assert t_path.name == "t_merged.npy"