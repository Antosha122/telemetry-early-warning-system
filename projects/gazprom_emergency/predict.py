"""Инференс (прогноз аварий) загруженной моделью.

Важно о гранулярности прогноза
==============================
Модель предсказывает аварийность **на уровне часа** (``batch_time``), а не
на уровне отдельной операции из ``opers.csv``.

В исходных данных одному ``batch_time`` соответствуют десятки операций на разных
установках (``r_id``). При подготовке обучающей выборки (``data.py``) метка
агрегируется как ``max(is_emergency)`` — если хоть одна операция в час аварийная,
весь час помечается как аварийный. Поэтому прогноз ``prediction=1`` для часа,
где «одна конкретная операция не аварийная», — это **не ложное срабатывание**:
в этом же часе могут быть другие аварийные операции.

Опция ``prediction.attach_opers_context`` (по умолчанию ``true``) добавляет
в выходной CSV столбцы из ``opers.csv``, чтобы это было видно напрямую:
``n_ops``, ``n_emergency_ops``, ``ground_truth``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, load_config
from .data import _feature_columns
from .model import load_checkpoint_metadata, load_model
from .utils import load_scaler_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _to_float_matrix(df, columns):
    """Конвертирует колонки в float32. Невалидные -> 0.0 (как Polars fill_null)."""
    result = np.empty((len(df), len(columns)), dtype=np.float32)
    for i, col in enumerate(columns):
        result[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
    return result


def extract_features_from_csv(df, feature_columns=None):
    """Извлекает матрицу признаков из DataFrame."""
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input CSV is missing {len(missing)} feature columns. "
                f"Model expects {len(feature_columns)} features."
            )
        return _to_float_matrix(df, feature_columns)
    feat_cols = _feature_columns({c: str for c in df.columns})
    return _to_float_matrix(df, feat_cols)


def _normalize_batch_time(series: pd.Series) -> pd.Series:
    """Приводит колонку времени к единому строковому формату для merge."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def load_opers_context(cfg: Config) -> pd.DataFrame | None:
    """Загружает и агрегирует opers.csv для контекста прогноза."""
    opers_path = Path(cfg.data.source_dir) / cfg.data.opers_file
    if not opers_path.exists():
        logger.warning("opers.csv not found at %s — context columns disabled", opers_path)
        return None

    opers_df = pd.read_csv(
        opers_path,
        sep=getattr(cfg.data, "opers_separator", ","),
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )

    time_col = cfg.data.opers_join_column
    if time_col not in opers_df.columns:
        logger.warning(
            "Column %r not found in opers.csv — context disabled", time_col
        )
        return None

    if "is_emergency" not in opers_df.columns:
        logger.warning("Column 'is_emergency' not found in opers.csv — context disabled")
        return None

    opers_df["_bt"] = _normalize_batch_time(opers_df[time_col])
    emergency = pd.to_numeric(opers_df["is_emergency"], errors="coerce").fillna(0).astype(int)

    context = (
        emergency.groupby(opers_df["_bt"])
        .agg(n_ops="size", n_emergency_ops="sum")
        .reset_index()
        .rename(columns={"_bt": "batch_time"})
    )
    context["ground_truth"] = (context["n_emergency_ops"] > 0).astype(int)
    logger.info(
        "Loaded opers context: %d timestamps, %d with emergencies",
        len(context),
        int(context["ground_truth"].sum()),
    )
    return context


def _attach_opers_context(result: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Присоединяет колонки контекста из opers.csv к результату прогноза."""
    if "batch_time" not in result.columns:
        logger.warning("No batch_time column in predictions — context disabled")
        return result

    context = load_opers_context(cfg)
    if context is None:
        return result

    result = result.copy()
    result["_bt_key"] = _normalize_batch_time(result["batch_time"])
    merged = result.merge(
        context, left_on="_bt_key", right_on="batch_time", how="left", suffixes=("", "_opers")
    )
    merged = merged.drop(columns=["_bt_key", "batch_time_opers"], errors="ignore")

    for col in ["n_ops", "n_emergency_ops", "ground_truth"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    logger.info(
        "Attached opers context: %d/%d predictions have ground_truth=1",
        int(merged["ground_truth"].sum()) if "ground_truth" in merged.columns else 0,
        len(merged),
    )
    return merged


def predict_batch(cfg_path, input_csv=None):
    """Делает предсказание."""
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_checkpoint_metadata(cfg.model.save_path)
    feature_columns = metadata.get("feature_columns")
    expected_features = metadata.get("input_dim") or metadata.get("n_features")

    # Приоритет порога: оптимизированный при обучении > cfg.prediction.threshold.
    # Это критически важно: при обучении порог подбирается под валидационную
    # выборку (например 0.03 для F1). Статический 0.78 из конфига почти
    # гарантированно неоптимален и приводит к нулевому recall.
    optimized_threshold = metadata.get("extra", {}).get("optimal_threshold")
    if optimized_threshold is not None and getattr(
        cfg.prediction, "use_optimized_threshold", True
    ):
        effective_threshold = float(optimized_threshold)
        logger.info(
            "Using optimized threshold %.4f from checkpoint (config has %.4f)",
            effective_threshold,
            cfg.prediction.threshold,
        )
    else:
        effective_threshold = cfg.prediction.threshold
        logger.info("Using threshold %.4f from config", effective_threshold)

    model = load_model(cfg.model.save_path).to(device)

    scaler_path = Path(cfg.model.save_path).with_suffix(".scaler.json")
    scaler = load_scaler_json(scaler_path) if scaler_path.exists() else None

    input_path = Path(input_csv) if input_csv else Path(cfg.data.source_dir) / cfg.data.stpa_file
    logger.info("Loading input: %s", input_path)

    df = pd.read_csv(
        input_path,
        sep=getattr(cfg.data, "stpa_separator", ";"),
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    if getattr(cfg.data, "stpa_skip_first_column", False):
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or c == ""]
        if unnamed:
            df = df.drop(columns=unnamed)

    batch_time_col = None
    for candidate in ["batch_time", "date", cfg.data.opers_join_column]:
        if candidate in df.columns:
            batch_time_col = candidate
            break
    batch_times = df[batch_time_col].values if batch_time_col else None

    X = extract_features_from_csv(df, feature_columns)

    if expected_features is not None and expected_features != X.shape[1]:
        raise ValueError(f"Dim mismatch: model={expected_features}, input={X.shape[1]}")

    if scaler is not None:
        X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy()

    preds = (probs >= effective_threshold).astype(int)

    result_data = {"probability": probs, "prediction": preds, "threshold": effective_threshold}
    if batch_times is not None:
        result_data["batch_time"] = batch_times
    result = pd.DataFrame(result_data)

    if getattr(cfg.prediction, "attach_opers_context", True):
        result = _attach_opers_context(result, cfg)

    result = result.sort_values("probability", ascending=False).reset_index(drop=True)

    output_path = Path(cfg.prediction.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    n_emergency = int(preds.sum())
    n_total = len(preds)
    logger.info("Predictions saved to %s (%d rows)", output_path, n_total)
    logger.info("=== Summary ===")
    logger.info("Total predictions:     %d", n_total)
    logger.info("Predicted emergencies: %d (%.1f%%)", n_emergency, 100.0 * n_emergency / max(n_total, 1))
    logger.info("Threshold:             %.4f", effective_threshold)

    if "ground_truth" in result.columns:
        gt = result["ground_truth"].values
        tp = int(((preds == 1) & (gt == 1)).sum())
        fp = int(((preds == 1) & (gt == 0)).sum())
        fn = int(((preds == 0) & (gt == 1)).sum())
        tn = int(((preds == 0) & (gt == 0)).sum())
        logger.info("=== Ground-truth comparison (hour-level) ===")
        logger.info("True positives:  %d", tp)
        logger.info("False positives: %d", fp)
        logger.info("False negatives: %d", fn)
        logger.info("True negatives:  %d", tn)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        logger.info("Precision: %.4f, Recall: %.4f", precision, recall)

    if batch_times is not None:
        high_risk = result[result["prediction"] == 1]
        if len(high_risk) > 0:
            logger.info("High-risk periods (top 10):")
            for _, row in high_risk.head(10).iterrows():
                gt_str = ""
                if "ground_truth" in row.index:
                    gt_str = f" | ground_truth={int(row['ground_truth'])}"
                    if "n_emergency_ops" in row.index:
                        gt_str += f" ({int(row['n_emergency_ops'])}/{int(row['n_ops'])} ops)"
                logger.info(
                    "  %s | probability=%.4f%s",
                    row.get("batch_time", "?"),
                    row["probability"],
                    gt_str,
                )
        else:
            top5 = result.head(5)
            logger.info("No emergencies predicted. Top 5 highest probabilities:")
            for _, row in top5.iterrows():
                logger.info("  %s | probability=%.4f", row.get("batch_time", "?"), row["probability"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()
    predict_batch(args.config, args.input)


if __name__ == "__main__":
    main()