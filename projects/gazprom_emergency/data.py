"""Потоковая загрузка и подготовка данных прогноза аварий.

Стратегия:
1. Исходные данные (opers.csv + stpa.csv) объединяются по batch_time.
2. Polars LazyFrame обеспечивает потоковую обработку без загрузки в ОЗУ.
3. Результат сохраняется в np.memmap для эффективного доступа во время обучения.
4. Список признаков (``feature_columns``) сохраняется в JSON рядом с memmap —
   единый источник правды для обучения и инференса (решение замечания #7).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from .config import DataConfig

logger = logging.getLogger(__name__)

# Ожидаемые имена колонок
COL_BATCH_TIME = "batch_time"
COL_IS_EMERGENCY = "is_emergency"
FEATURE_PREFIX = "v_"

# Имя JSON-файла со списком признаков (сохраняется рядом с memmap)
FEATURE_COLUMNS_FILE = "feature_columns.json"


def _feature_columns(schema: dict[str, type]) -> list[str]:
    """Возвращает список признаков v_* из схемы."""
    return sorted(
        [c for c in schema if c.startswith(FEATURE_PREFIX)],
        key=lambda c: int(c.split("_")[1]),
    )


def merge_to_memmap(cfg: DataConfig) -> tuple[Path, Path, int]:
    """Объединяет opers + stpa и сохраняет в memmap-файлы.

    Дополнительно сохраняет массив batch_time в отдельный memmap-файл
    (``t_merged.npy``) для последующего хронологического разбиения train/val.

    Returns:
        (x_path, y_path, n_features) — пути к .npy memmap и число признаков.
    """
    source_dir = Path(cfg.source_dir)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    opers_path = source_dir / cfg.opers_file
    stpa_path = source_dir / cfg.stpa_file

    if not opers_path.exists():
        raise FileNotFoundError(f"opers file not found: {opers_path}")
    if not stpa_path.exists():
        raise FileNotFoundError(f"stpa file not found: {stpa_path}")

    # --- Lazy load через Polars ---
    logger.info("Scanning %s and %s", opers_path, stpa_path)
    opers = pl.scan_csv(opers_path, try_parse_dates=True)
    stpa = pl.scan_csv(stpa_path, try_parse_dates=True)

    # Определяем число признаков по схеме stpa
    stpa_schema = stpa.collect_schema()
    feat_cols = _feature_columns(dict(stpa_schema))
    n_features = len(feat_cols)
    logger.info("Found %d feature columns (v_0 .. v_%d)", n_features, n_features - 1)

    # Join: по batch_time берём is_emergency из opers и все v_* из stpa
    join_col = COL_BATCH_TIME
    merged = stpa.join(opers.select([join_col, COL_IS_EMERGENCY]), on=join_col, how="inner")

    # --- Первый проход: определяем число строк ---
    logger.info("Counting rows (streaming)...")
    n_rows = merged.select(pl.len()).collect(engine="streaming").item()
    logger.info("Total rows: %d", n_rows)

    x_path = processed_dir / "X_merged.npy"
    y_path = processed_dir / "y_merged.npy"
    t_path = processed_dir / "t_merged.npy"

    # Создаём memmap-файлы (X, y и временные метки batch_time)
    X_mm = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, n_features))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(n_rows,))
    # batch_time храним как int64 наносекунд (datetime64[ns]) для сортировки
    t_mm = np.memmap(t_path, dtype=np.int64, mode="w+", shape=(n_rows,))

    # --- Второй проход: собираем данные ---
    logger.info("Writing memmap (streaming)...")
    offset = 0
    select_cols = feat_cols + [COL_IS_EMERGENCY, COL_BATCH_TIME]

    # Стриминговый collect (Polars 1.x не поддерживает chunk_size в collect)
    batches = merged.select(select_cols).collect(engine="streaming")

    # Polars может вернуть один DataFrame или несколько; нормализуем
    if isinstance(batches, pl.DataFrame):
        batches = [batches]

    for batch in tqdm(batches, desc="Processing batches"):
        rows = batch.height
        feat_arr = batch.select(feat_cols).to_numpy().astype(np.float32)
        target_arr = batch.select(COL_IS_EMERGENCY).to_numpy().ravel().astype(np.float32)
        time_arr = batch.select(COL_BATCH_TIME).to_numpy().ravel().astype("datetime64[ns]").astype(np.int64)

        X_mm[offset : offset + rows] = feat_arr
        y_mm[offset : offset + rows] = target_arr
        t_mm[offset : offset + rows] = time_arr
        offset += rows

    X_mm.flush()
    y_mm.flush()
    t_mm.flush()
    logger.info("Wrote %d rows to %s, %s and %s", offset, x_path, y_path, t_path)

    return x_path, y_path, n_features


def load_memmap(
    x_path: str | Path, y_path: str | Path, n_features: int
) -> tuple[np.memmap, np.memmap]:
    """Открывает memmap-файлы признаков и целевой для чтения."""
    x_path = Path(x_path)
    y_path = Path(y_path)

    # Размер файла / (4 байта * n_features) = число строк
    n_rows = x_path.stat().st_size // (4 * n_features)

    X = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n_rows, n_features))
    y = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_rows,))
    return X, y


def load_batch_times(t_path: str | Path) -> np.ndarray:
    """Открывает memmap-файл временных меток batch_time для чтения.

    Возвращает массив ``datetime64[ns]`` для хронологической сортировки/разбиения.
    """
    t_path = Path(t_path)
    if not t_path.exists():
        raise FileNotFoundError(f"batch_time memmap not found: {t_path}")
    n_rows = t_path.stat().st_size // 8  # int64 = 8 байт
    t_mm = np.memmap(t_path, dtype=np.int64, mode="r", shape=(n_rows,))
    return t_mm.astype("datetime64[ns]")


def save_feature_columns(columns: list[str], path: str | Path) -> None:
    """Сохраняет упорядоченный список имён признаков в JSON.

    Это единый источник правды: и train, и predict используют один список,
    что исключает рассинхрон (решение замечания #7 senior review).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(columns), "n_features": len(columns)}, f)
    logger.info("Feature columns (%d) saved to %s", len(columns), path)


def load_feature_columns(path: str | Path) -> list[str]:
    """Загружает список имён признаков из JSON.

    Args:
        path: путь к ``feature_columns.json``.

    Returns:
        Упорядоченный список имён признаков.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["feature_columns"])


def get_feature_columns_path(cfg: DataConfig) -> Path:
    """Возвращает путь к JSON-файлу со списком признаков."""
    return Path(cfg.processed_dir) / FEATURE_COLUMNS_FILE


def get_processed_paths(cfg: DataConfig) -> tuple[Path, Path, Path]:
    """Возвращает пути к уже обработанным memmap-файлам (X, y, batch_time)."""
    processed_dir = Path(cfg.processed_dir)
    return (
        processed_dir / "X_merged.npy",
        processed_dir / "y_merged.npy",
        processed_dir / "t_merged.npy",
    )
