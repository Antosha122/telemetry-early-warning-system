"""Тесты для predict.py: контекст opers, нормализация времени."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.gazprom_emergency.config import Config, DataConfig
from projects.gazprom_emergency.predict import (
    _attach_opers_context,
    _normalize_batch_time,
    extract_features_from_csv,
    load_opers_context,
)


# ============================================================
# _normalize_batch_time
# ============================================================

class TestNormalizeBatchTime:
    """Тестирует нормализацию форматов времени для merge."""

    def test_strips_microseconds(self):
        """Формат с микросекундами нормализуется (убираются .000000)."""
        s = pd.Series(["2023-09-26 07:00:00.000000"])
        result = _normalize_batch_time(s)
        assert result.iloc[0] == "2023-09-26 07:00:00"

    def test_different_formats_match(self):
        """Разные форматы одной даты дают одинаковую нормализованную строку."""
        s = pd.Series(
            ["2023-09-26 07:00:00.000000", "2023-09-26 07:00:00", "2023-09-26T07:00:00"]
        )
        result = _normalize_batch_time(s)
        assert result.nunique() == 1
        assert result.iloc[0] == "2023-09-26 07:00:00"

    def test_invalid_returns_nat(self):
        """Невалидная строка -> NaT."""
        s = pd.Series(["not-a-date"])
        result = _normalize_batch_time(s)
        assert pd.isna(result.iloc[0])


# ============================================================
# load_opers_context
# ============================================================

class TestLoadOpersContext:
    """Тестирует загрузку и агрегацию opers.csv для контекста прогноза."""

    def _write_opers(self, path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_aggregates_emergency_max(self, tmp_path: Path):
        """Час с хотя бы одной аварийной операцией → ground_truth=1."""
        self._write_opers(
            tmp_path / "opers.csv",
            [
                {"date": "2023-01-01 00:00:00", "is_emergency": False},
                {"date": "2023-01-01 00:00:00", "is_emergency": True},
                {"date": "2023-01-01 01:00:00", "is_emergency": False},
            ],
        )
        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="opers.csv"))
        ctx = load_opers_context(cfg)
        assert ctx is not None
        row = ctx[ctx["batch_time"] == "2023-01-01 00:00:00"].iloc[0]
        assert row["n_ops"] == 2
        assert row["n_emergency_ops"] == 1
        assert row["ground_truth"] == 1

    def test_all_false_hour(self, tmp_path: Path):
        """Час без аварийных операций → ground_truth=0."""
        self._write_opers(
            tmp_path / "opers.csv",
            [{"date": "2023-01-01 00:00:00.000000", "is_emergency": False}],
        )
        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="opers.csv"))
        ctx = load_opers_context(cfg)
        assert ctx is not None
        row = ctx[ctx["batch_time"] == "2023-01-01 00:00:00"].iloc[0]
        assert row["ground_truth"] == 0

    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """Нет opers.csv → None (контекст выключается)."""
        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="missing.csv"))
        assert load_opers_context(cfg) is None

    def test_handles_numeric_emergency(self, tmp_path: Path):
        """is_emergency может быть числом — обрабатывается корректно."""
        self._write_opers(
            tmp_path / "opers.csv",
            [
                {"date": "2023-01-01 00:00:00", "is_emergency": 1},
                {"date": "2023-01-01 00:00:00", "is_emergency": 0},
            ],
        )
        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="opers.csv"))
        ctx = load_opers_context(cfg)
        assert ctx is not None
        row = ctx.iloc[0]
        assert row["n_emergency_ops"] == 1
        assert row["ground_truth"] == 1


# ============================================================
# _attach_opers_context
# ============================================================

class TestAttachOpersContext:
    """Тестирует присоединение контекста к прогнозу."""

    def test_context_attached(self, tmp_path: Path):
        """К прогнозу добавляются n_ops, n_emergency_ops, ground_truth."""
        pd.DataFrame(
            [
                {"date": "2023-01-01 00:00:00", "is_emergency": True},
                {"date": "2023-01-01 00:00:00", "is_emergency": False},
                {"date": "2023-01-01 01:00:00", "is_emergency": False},
            ]
        ).to_csv(tmp_path / "opers.csv", index=False)

        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="opers.csv"))
        result = pd.DataFrame(
            {
                "probability": [0.99, 0.10],
                "prediction": [1, 0],
                "batch_time": [
                    "2023-01-01 00:00:00",
                    "2023-01-01 01:00:00",
                ],
            }
        )
        merged = _attach_opers_context(result, cfg)
        assert "ground_truth" in merged.columns
        assert "n_ops" in merged.columns
        assert "n_emergency_ops" in merged.columns

        row0 = merged[merged["batch_time"] == "2023-01-01 00:00:00"].iloc[0]
        assert row0["ground_truth"] == 1
        assert row0["n_emergency_ops"] == 1
        assert row0["n_ops"] == 2

        row1 = merged[merged["batch_time"] == "2023-01-01 01:00:00"].iloc[0]
        assert row1["ground_truth"] == 0

    def test_missing_opers_returns_original(self, tmp_path: Path):
        """Без opers.csv результат возвращается без новых колонок."""
        cfg = Config(data=DataConfig(source_dir=str(tmp_path), opers_file="missing.csv"))
        result = pd.DataFrame(
            {"probability": [0.5], "prediction": [1], "batch_time": ["2023-01-01"]}
        )
        merged = _attach_opers_context(result, cfg)
        assert "ground_truth" not in merged.columns


# ============================================================
# extract_features_from_csv
# ============================================================

class TestExtractFeatures:
    """Тестирует извлечение признаков из DataFrame."""

    def test_valid_floats(self):
        """Корректные float колонки извлекаются как float32."""
        df = pd.DataFrame({"v_0": [1.0, 2.0], "v_1": [3.0, 4.0]})
        X = extract_features_from_csv(df, ["v_0", "v_1"])
        assert X.shape == (2, 2)
        assert X.dtype == np.float32

    def test_invalid_become_zero(self):
        """Невалидные значения заменяются на 0.0."""
        df = pd.DataFrame({"v_0": ["1.0", "abc", ""]})
        X = extract_features_from_csv(df, ["v_0"])
        assert X[1, 0] == 0.0

    def test_missing_column_raises(self):
        """Отсутствующая колонка вызывает ValueError."""
        df = pd.DataFrame({"v_0": [1.0]})
        with pytest.raises(ValueError, match="missing 1 feature columns"):
            extract_features_from_csv(df, ["v_0", "v_1"])