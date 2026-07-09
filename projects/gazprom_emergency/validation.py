"""Валидация входных данных (замечание senior review #11).

Проверяет и очищает входные CSV перед merge:
- дубликаты ключей ``batch_time`` в opers.csv,
- NaN/inf в фичах,
- корректность типов ``v_*`` колонок,
- диапазон целевой ``is_emergency`` ∈ {0, 1},
- защита от many-to-many join (размножение строк при неуникальных ключах).

Все стратегии настраиваются через ``ValidationConfig``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config import ValidationConfig

logger = logging.getLogger(__name__)

COL_BATCH_TIME = "batch_time"
COL_IS_EMERGENCY = "is_emergency"
FEATURE_PREFIX = "v_"


@dataclass
class ValidationReport:
    """Отчёт о результатах валидации."""

    opers_duplicates_removed: int = 0
    nan_filled: int = 0
    nan_rows_dropped: int = 0
    inf_replaced: int = 0
    invalid_target_rows: int = 0
    join_multiplicity: float = 1.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def __str__(self) -> str:
        lines = ["=== Validation Report ==="]
        lines.append(f"  opers duplicates removed: {self.opers_duplicates_removed}")
        lines.append(f"  NaN filled:               {self.nan_filled}")
        lines.append(f"  NaN rows dropped:         {self.nan_rows_dropped}")
        lines.append(f"  inf replaced:             {self.inf_replaced}")
        lines.append(f"  invalid target rows:      {self.invalid_target_rows}")
        lines.append(f"  join multiplicity:        {self.join_multiplicity:.2f}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


def _feature_columns(schema: dict[str, type]) -> list[str]:
    """Возвращает список признаков v_* из схемы."""
    return sorted(
        [c for c in schema if c.startswith(FEATURE_PREFIX)],
        key=lambda c: int(c.split("_")[1]),
    )


def validate_opers(
    opers: pl.DataFrame, cfg: ValidationConfig, report: ValidationReport
) -> pl.DataFrame:
    """Валидирует opers DataFrame: дубликаты ключей, целевая переменная.

    Защита от many-to-many join: если ``batch_time`` в opers не уникален,
    inner join с stpa размножит строки. Стратегии:
    - ``"fail"`` — поднять ValueError (по умолчанию, safest).
    - ``"drop"`` — оставить только первое вхождение.
    - ``"aggregate"`` — взять максимум ``is_emergency`` по дубликатам
      (если хоть одна запись помечена аварийной — считаем аварийой).
    """
    if COL_BATCH_TIME not in opers.columns:
        report.errors.append(f"Column {COL_BATCH_TIME!r} not found in opers")
        return opers

    # Проверка целевой переменной
    if cfg.strict_binary_target and COL_IS_EMERGENCY in opers.columns:
        target = opers.select(COL_IS_EMERGENCY).to_series()
        unique_vals = set(target.unique().to_list())
        invalid = unique_vals - {0, 1, False, True, 0.0, 1.0}
        if invalid:
            report.invalid_target_rows = int(
                (~target.is_in([0, 1])).sum()
            )
            if report.invalid_target_rows > 0:
                report.errors.append(
                    f"Target {COL_IS_EMERGENCY!r} has invalid values: {invalid}. "
                    f"Expected only {{0, 1}}."
                )

    # Дубликаты batch_time
    dup_count = int(opers.select(COL_BATCH_TIME).to_series().is_duplicated().sum())
    if dup_count == 0:
        return opers

    strategy = cfg.duplicate_keys_strategy
    if strategy == "fail":
        report.errors.append(
            f"Found {dup_count} duplicate keys in opers.{COL_BATCH_TIME}. "
            f"Strategy='fail'. Use duplicate_keys_strategy='drop' or 'aggregate'."
        )
    elif strategy == "drop":
        report.opers_duplicates_removed = dup_count
        opers = opers.unique(subset=[COL_BATCH_TIME], keep="first")
        report.warnings.append(
            f"Removed {dup_count} duplicate {COL_BATCH_TIME} rows (keep=first)"
        )
    elif strategy == "aggregate":
        report.opers_duplicates_removed = dup_count
        opers = opers.group_by(COL_BATCH_TIME).agg(
            pl.col(COL_IS_EMERGENCY).max()
        )
        report.warnings.append(
            f"Aggregated {dup_count} duplicate {COL_BATCH_TIME} rows "
            f"(is_emergency=max)"
        )
    else:
        report.errors.append(f"Unknown duplicate_keys_strategy: {strategy!r}")

    return opers


def validate_stpa(
    stpa: pl.DataFrame, cfg: ValidationConfig, report: ValidationReport
) -> pl.DataFrame:
    """Валидирует stpa DataFrame: NaN/inf в фичах, типы колонок.

    Применяет стратегии из ``ValidationConfig``:
    - ``nan_strategy``: как обрабатывать NaN.
    - ``inf_strategy``: как обрабатывать inf.
    - ``max_nan_fraction_per_row``: доля NaN, при которой строка отбрасывается.
    """
    feat_cols = _feature_columns(dict(stpa.schema))
    if not feat_cols:
        report.errors.append("No feature columns (v_*) found in stpa")
        return stpa

    # Приведение типов
    expected_dtype = (
        pl.Float32 if cfg.expected_feature_dtype == "float32" else pl.Float64
    )
    cast_exprs = [pl.col(c).cast(expected_dtype) for c in feat_cols]
    stpa = stpa.with_columns(cast_exprs)

    # Обработка inf
    if cfg.inf_strategy == "fail":
        # Проверяем, есть ли inf
        for col in feat_cols:
            col_data = stpa.select(pl.col(col).abs().is_infinite()).to_series()
            inf_count = int(col_data.sum())
            if inf_count > 0:
                report.errors.append(
                    f"Found {inf_count} inf values in {col!r}. "
                    f"inf_strategy='fail'."
                )
                break
        else:
            report.inf_replaced = 0
    elif cfg.inf_strategy == "replace_with_nan":
        total_inf = 0
        replace_exprs = []
        for col in feat_cols:
            replace_exprs.append(
                pl.when(pl.col(col).abs() == float("inf"))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            total_inf += int(
                stpa.select(pl.col(col).abs().is_infinite()).to_series().sum()
            )
        if total_inf > 0:
            stpa = stpa.with_columns(replace_exprs)
            report.inf_replaced = total_inf
            report.warnings.append(
                f"Replaced {total_inf} inf values with NaN (will be handled by nan_strategy)"
            )

    # Подсчёт NaN по строкам
    n_features = len(feat_cols)
    nan_per_row = stpa.select(
        pl.sum_horizontal([pl.col(c).is_null() for c in feat_cols])
    ).to_series()

    # Отбрасывание строк с слишком большим количеством NaN
    nan_fraction = nan_per_row / n_features
    rows_to_drop_mask = nan_fraction > cfg.max_nan_fraction_per_row
    rows_to_drop = int(rows_to_drop_mask.sum())
    if rows_to_drop > 0:
        report.nan_rows_dropped = rows_to_drop
        stpa = stpa.filter(~rows_to_drop_mask)
        report.warnings.append(
            f"Dropped {rows_to_drop} rows with >{cfg.max_nan_fraction_per_row:.0%} NaN"
        )

    # Обработка оставшихся NaN
    if cfg.nan_strategy == "fail":
        total_nan = int(
            stpa.select(
                pl.sum_horizontal([pl.col(c).is_null() for c in feat_cols])
            )
            .to_series()
            .sum()
        )
        if total_nan > 0:
            report.errors.append(
                f"Found {total_nan} NaN in features. nan_strategy='fail'."
            )
    elif cfg.nan_strategy == "drop":
        total_nan = int(
            stpa.select(
                pl.sum_horizontal([pl.col(c).is_null() for c in feat_cols])
            ).to_series()
            .sum()
        )
        if total_nan > 0:
            stpa = stpa.drop_nulls(subset=feat_cols)
            report.nan_filled = total_nan
            report.warnings.append(f"Dropped rows with NaN ({total_nan} cells)")
    elif cfg.nan_strategy in ("fill_mean", "fill_median", "fill_zero"):
        fill_exprs = []
        total_filled = 0
        for col in feat_cols:
            col_data = stpa.select(pl.col(col))
            nan_count = int(col_data.to_series().is_null().sum())
            if nan_count == 0:
                continue
            total_filled += nan_count
            if cfg.nan_strategy == "fill_zero":
                fill_exprs.append(pl.col(col).fill_null(0.0))
            elif cfg.nan_strategy == "fill_mean":
                mean_val = float(col_data.to_series().mean() or 0.0)
                fill_exprs.append(pl.col(col).fill_null(mean_val))
            elif cfg.nan_strategy == "fill_median":
                median_val = float(col_data.to_series().median() or 0.0)
                fill_exprs.append(pl.col(col).fill_null(median_val))
        if fill_exprs:
            stpa = stpa.with_columns(fill_exprs)
            report.nan_filled = total_filled
            report.warnings.append(
                f"Filled {total_filled} NaN with {cfg.nan_strategy}"
            )
    else:
        report.errors.append(f"Unknown nan_strategy: {cfg.nan_strategy!r}")

    return stpa


def check_join_multiplicity(
    opers: pl.DataFrame,
    stpa: pl.DataFrame,
    cfg: ValidationConfig,
    report: ValidationReport,
) -> None:
    """Проверяет, не приведёт ли join к размножению строк.

    Защита от many-to-many join: если в обеих таблицах есть дубликаты
    по ``batch_time``, inner join создаст декартово произведение дубликатов.
    """
    opers_keys = opers.select(COL_BATCH_TIME).to_series()
    stpa_keys = stpa.select(COL_BATCH_TIME).to_series()

    opers_n = len(opers)
    stpa_n = len(stpa)

    # Оценка: для каждого ключа считаем min(count_opers, count_stpa)
    opers_counts = opers.group_by(COL_BATCH_TIME).len().rename({"len": "opers_n"})
    stpa_counts = stpa.group_by(COL_BATCH_TIME).len().rename({"len": "stpa_n"})
    merged_counts = opers_counts.join(stpa_counts, on=COL_BATCH_TIME, how="inner")
    estimated_rows = int(
        merged_counts.select(
            pl.min_horizontal(["opers_n", "stpa_n"]).sum()
        ).item()
    )

    # Дополнительно: средняя кратность
    max_key_n = max(
        int(opers_counts.select("opers_n").max() or 1),
        int(stpa_counts.select("stpa_n").max() or 1),
    )

    # Коэффициент размножения относительно минимально ожидаемого
    expected_unique = int(
        merged_counts.select(pl.len()).item()
    )
    multiplicity = estimated_rows / max(expected_unique, 1)
    report.join_multiplicity = float(multiplicity)

    if max_key_n > 1:
        report.warnings.append(
            f"Non-unique keys detected: opers max={int(opers_counts.select('opers_n').max() or 1)}, "
            f"stpa max={int(stpa_counts.select('stpa_n').max() or 1)}. "
            f"Estimated join rows: {estimated_rows} (multiplicity={multiplicity:.2f})."
        )

    if multiplicity > cfg.max_join_multiplicity:
        report.errors.append(
            f"Join multiplicity {multiplicity:.2f} exceeds threshold "
            f"{cfg.max_join_multiplicity:.2f}. This indicates a many-to-many "
            f"join that will duplicate rows. Ensure batch_time is unique in opers.csv."
        )


def validate_inputs(
    opers: pl.DataFrame,
    stpa: pl.DataFrame,
    cfg: ValidationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, ValidationReport]:
    """Полная валидация входных данных перед merge.

    Args:
        opers: DataFrame с колонками ``batch_time``, ``is_emergency``.
        stpa: DataFrame с колонками ``batch_time``, ``v_*``.
        cfg: параметры валидации.

    Returns:
        Кортеж ``(opers_clean, stpa_clean, report)``.

    Raises:
        ValueError: если стратегия "fail" сработала на найденных проблемах
            или конфигурация невалидна.
    """
    report = ValidationReport()

    if not cfg.enabled:
        report.warnings.append("Validation disabled (ValidationConfig.enabled=False)")
        return opers, stpa, report

    # 1. Валидация opers
    opers_clean = validate_opers(opers, cfg, report)

    # 2. Валидация stpa
    stpa_clean = validate_stpa(stpa, cfg, report)

    # 3. Проверка many-to-many join
    check_join_multiplicity(opers_clean, stpa_clean, cfg, report)

    logger.info("\n%s", str(report))

    if report.has_errors:
        raise ValueError(
            f"Validation failed with {len(report.errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in report.errors)
        )

    return opers_clean, stpa_clean, report


def validate_numpy_features(
    X: np.ndarray, cfg: ValidationConfig
) -> tuple[np.ndarray, ValidationReport]:
    """Валидирует numpy-матрицу признаков (NaN/inf).

    Используется после загрузки memmap для проверки целостности данных.

    Args:
        X: матрица признаков (n_rows, n_features).
        cfg: параметры валидации.

    Returns:
        Кортеж ``(X_clean, report)``.
    """
    report = ValidationReport()

    # inf
    inf_mask = ~np.isfinite(X) & (np.abs(X) != np.inf)
    inf_count = int(np.isinf(X).sum())
    if inf_count > 0:
        if cfg.inf_strategy == "fail":
            report.errors.append(f"Found {inf_count} inf in feature matrix")
        elif cfg.inf_strategy == "replace_with_nan":
            X = np.where(np.isinf(X), np.nan, X)
            report.inf_replaced = inf_count

    # nan
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        if cfg.nan_strategy == "fail":
            report.errors.append(f"Found {nan_count} NaN in feature matrix")
        elif cfg.nan_strategy == "fill_zero":
            X = np.nan_to_num(X, nan=0.0)
            report.nan_filled = nan_count
        elif cfg.nan_strategy == "fill_mean":
            col_means = np.nanmean(X, axis=0)
            # Заменяем nan-mean на 0 (если вся колонка nan)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            report.nan_filled = nan_count
        elif cfg.nan_strategy == "fill_median":
            # np.nanmedian может выдать предупреждение на all-nan slice
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_medians = np.nanmedian(X, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
            report.nan_filled = nan_count

    return X, report