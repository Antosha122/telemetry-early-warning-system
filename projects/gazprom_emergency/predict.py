"""Инференс (прогноз аварий) загруженной моделью.

Решение замечания #7 senior review:
- Список признаков (``feature_columns``) берётся из чекпоинта модели, а не
  извлекается заново из входного CSV — исключается рассинхрон между обучением
  и инференсом.
- Размерность входа (``input_dim`` / ``n_features``) — из метаданных чекпоинта,
  без хардкода.
- Строгая проверка: колонки в CSV должны точно совпадать с сохранёнными.
"""

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


def extract_features_from_csv(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> np.ndarray:
    """Извлекает матрицу признаков из DataFrame.

    Если ``feature_columns`` передан (из чекпоинта), используется он —
    это гарантирует тот же порядок и состав, что при обучении.
    Иначе — fallback: извлечение v_* из колонок CSV (с предупреждением).

    Args:
        df: входной DataFrame.
        feature_columns: упорядоченный список имён признаков из чекпоинта.

    Returns:
        Матрица признаков (n_rows, n_features) как float32.

    Raises:
        ValueError: если сохранённые feature_columns не найдены в CSV.
    """
    if feature_columns is not None:
        # Строгий путь: используем список из чекпоинта
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input CSV is missing {len(missing)} feature columns "
                f"expected by the model (e.g. {missing[:5]}). "
                f"Model was trained on {len(feature_columns)} features. "
                f"Ensure the input CSV matches the training data schema."
            )
        X = df[feature_columns].values.astype(np.float32)
        logger.info(
            "Extracted %d features from checkpoint feature_columns", len(feature_columns)
        )
    else:
        # Fallback: извлекаем v_* из колонок (обратная совместимость)
        logger.warning(
            "feature_columns not found in checkpoint — extracting v_* from CSV. "
            "This may cause train/inference desync. Re-train to embed feature_columns."
        )
        feat_cols = _feature_columns({c: str for c in df.columns})
        X = df[feat_cols].values.astype(np.float32)

    return X


def predict_batch(cfg_path: str, input_csv: str | None = None) -> pd.DataFrame:
    """Делает предсказание по батчу данных.

    Безопасная загрузка артефактов:
    - Модель: weights_only=True (нет RCE-риска).
    - Scaler: JSON (вместо pickle).
    - Архитектура и feature_columns: из метаданных чекпоинта (без хардкода).
    """
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем метаданные чекпоинта (feature_columns, input_dim, n_features)
    metadata = load_checkpoint_metadata(cfg.model.save_path)
    feature_columns = metadata.get("feature_columns")
    expected_features = metadata.get("input_dim") or metadata.get("n_features")

    # Загружаем модель (input_dim и cfg извлекаются из чекпоинта)
    model = load_model(cfg.model.save_path).to(device)

    # Загружаем scaler (JSON, безопасно)
    scaler_path = Path(cfg.model.save_path).with_suffix(".scaler.json")
    if scaler_path.exists():
        scaler = load_scaler_json(scaler_path)
    else:
        scaler = None
        logger.warning("Scaler file not found: %s — skipping scaling", scaler_path)

    # Загружаем входные данные
    input_path = Path(input_csv) if input_csv else Path(cfg.data.source_dir) / cfg.data.stpa_file
    logger.info("Loading input: %s", input_path)

    df = pd.read_csv(input_path)
    X = extract_features_from_csv(df, feature_columns)

    # Проверяем совместимость размерности с моделью
    if expected_features is not None and expected_features != X.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_features}, "
            f"but input has {X.shape[1]} features."
        )

    if scaler:
        X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= cfg.prediction.threshold).astype(int)

    result = pd.DataFrame(
        {
            "probability": probs,
            "prediction": preds,
            "threshold": cfg.prediction.threshold,
        }
    )

    output_path = Path(cfg.prediction.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", output_path, len(result))

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict emergencies")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input CSV for prediction")
    args = parser.parse_args()
    predict_batch(args.config, args.input)


if __name__ == "__main__":
    main()