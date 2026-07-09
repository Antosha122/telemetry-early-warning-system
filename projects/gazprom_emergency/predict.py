"""Инференс (прогноз аварий) загруженной моделью."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
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


def predict_batch(cfg_path, input_csv=None):
    """Делает предсказание."""
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_checkpoint_metadata(cfg.model.save_path)
    feature_columns = metadata.get("feature_columns")
    expected_features = metadata.get("input_dim") or metadata.get("n_features")

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

    # Сохраняем batch_time для вывода (когда возможна авария)
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

    preds = (probs >= cfg.prediction.threshold).astype(int)

    result_data = {"probability": probs, "prediction": preds, "threshold": cfg.prediction.threshold}
    if batch_times is not None:
        result_data["batch_time"] = batch_times
    result = pd.DataFrame(result_data)

    # Сортируем по вероятности (самые опасные — сверху)
    result = result.sort_values("probability", ascending=False).reset_index(drop=True)

    output_path = Path(cfg.prediction.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    # Информативная сводка
    n_emergency = int(preds.sum())
    n_total = len(preds)
    logger.info("Predictions saved to %s (%d rows)", output_path, n_total)
    logger.info("=== Summary ===")
    logger.info("Total predictions:     %d", n_total)
    logger.info("Predicted emergencies: %d (%.1f%%)", n_emergency, 100.0 * n_emergency / max(n_total, 1))
    logger.info("Threshold:             %.4f", cfg.prediction.threshold)
    if batch_times is not None:
        high_risk = result[result["prediction"] == 1]
        if len(high_risk) > 0:
            logger.info("High-risk periods (top 10):")
            for _, row in high_risk.head(10).iterrows():
                logger.info("  %s | probability=%.4f", row.get("batch_time", "?"), row["probability"])
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
