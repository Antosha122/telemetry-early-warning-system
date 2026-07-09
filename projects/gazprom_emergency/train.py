"""Обучение модели прогноза аварий.

Аритектурные принципы (исправление замечаний senior review):
- Хронологическое разбиение по batch_time (без data leakage).
- Работа поверх np.memmap по индексам (без загрузки всех данных в ОЗУ).
- Инкрементальный fit scaler'а (чанками) — StandardScaler или MinMaxScaler.
- Фиксация всех seed'ов (torch, numpy, random, cudnn).
- Безопасная сериализация артефактов (weights_only=True, JSON scaler).
- Модульная структура: каждая функция тестируема независимо.
- Оптимизаторы и модели через реестр (Registry Pattern).
- Cost-sensitive learning через pos_weight (#17) без двойной компенсации.
- Gradient clipping и LR scheduler для стабильности обучения.
- Предвычисление train-данных в память для скорости обучения.
- Оптимизация порога после обучения (#14).
- MLflow трекинг (#15).
- Сохранение конфига рядом с моделью (#15).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    FEATURE_COLUMNS_FILE,
    get_feature_columns_path,
    load_batch_times,
    load_feature_columns,
    load_memmap,
    merge_to_memmap,
)
from .dataset import MemmapDataset, chronological_split_indices
from .mlflow_tracker import MLflowTracker, save_config_artifact
from .model import build_model, save_model
from .optimizers import build_optimizer
from .threshold import optimize_threshold_pipeline
from .utils import fit_scaler, save_scaler_json, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. Подготовка данных
# ============================================================

def prepare_data(cfg: Config) -> dict[str, Any]:
    """Загружает/создаёт memmap-данные и возвращает пути и размерности."""
    from .data import get_processed_paths

    x_path, y_path, t_path = get_processed_paths(cfg.data)
    fc_path = get_feature_columns_path(cfg.data)

    if not x_path.exists():
        logger.info("Processed data not found, running merge_to_memmap...")
        x_path, y_path, n_features = merge_to_memmap(cfg.data)
        n_rows = _count_rows(y_path)
    else:
        n_rows, n_features = _infer_shape(x_path)

    if fc_path.exists():
        feature_columns = load_feature_columns(fc_path)
    else:
        logger.warning(
            "%s not found, generating feature names from n_features=%d",
            FEATURE_COLUMNS_FILE,
            n_features,
        )
        feature_columns = [f"v_{i}" for i in range(n_features)]

    if len(feature_columns) != n_features:
        raise ValueError(
            f"Feature columns count mismatch: feature_columns.json has "
            f"{len(feature_columns)}, but memmap has {n_features} features. "
            f"Re-run merge_to_memmap to regenerate."
        )

    logger.info(
        "Data ready: x_path=%s, n_features=%d, n_rows=%d",
        x_path,
        n_features,
        n_rows,
    )
    return {
        "x_path": x_path,
        "y_path": y_path,
        "t_path": t_path,
        "n_features": n_features,
        "n_rows": n_rows,
        "feature_columns": feature_columns,
    }


def _infer_shape(x_path: Path) -> tuple[int, int]:
    """Определяет (n_rows, n_features) из размера файла X memmap."""
    n_rows = _count_rows(x_path.parent / "y_merged.npy")
    n_features = x_path.stat().st_size // (4 * n_rows)
    return n_rows, n_features


def _count_rows(y_path: Path) -> int:
    """Определяет число строк по размеру y memmap."""
    return y_path.stat().st_size // 4  # float32 = 4 байта


# ============================================================
# 2. Разбиение train/val
# ============================================================

def split_train_val(
    cfg: Config, data_info: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Разбивает индексы на train/val по выбранной стратегии."""
    x_path = data_info["x_path"]
    y_path = data_info["y_path"]
    n_features = data_info["n_features"]
    t_path = data_info["t_path"]

    X_mm, y_mm = load_memmap(x_path, y_path, n_features)
    n_rows = X_mm.shape[0]

    if cfg.training.split_strategy == "chronological":
        batch_times = load_batch_times(t_path)
        train_idx, val_idx = chronological_split_indices(
            batch_times, cfg.training.test_size
        )
    elif cfg.training.split_strategy == "random":
        logger.warning(
            "Using random split — risk of data leakage for time-series data!"
        )
        all_idx = np.arange(n_rows)
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_state,
        )
    else:
        raise ValueError(f"Unknown split_strategy: {cfg.training.split_strategy!r}")

    logger.info(
        "Split: train=%d (%.1f%%), val=%d (%.1f%%)",
        len(train_idx),
        100 * len(train_idx) / n_rows,
        len(val_idx),
        100 * len(val_idx) / n_rows,
    )
    return train_idx, val_idx


# ============================================================
# 3. SMOTE (балансировка классов)
# ============================================================

def apply_smote(
    X: np.memmap,
    y: np.memmap,
    train_idx: np.ndarray,
    cfg: Config,
    scaler: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет SMOTE к обучающей выборке (с учётом scaler)."""
    logger.info("Applying SMOTE (sampling_strategy=%.2f)...", cfg.training.smote_sampling_strategy)
    smote = SMOTE(
        sampling_strategy=cfg.training.smote_sampling_strategy,
        random_state=cfg.training.random_state,
    )

    X_train = _load_train_chunked(X, train_idx, scaler, cfg.data.chunk_size)
    y_train = np.asarray(y[train_idx], dtype=np.float32)

    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info("After SMOTE: X_train=%s (was %d)", X_res.shape, len(train_idx))
    return X_res, y_res


def _load_train_chunked(
    X: np.memmap,
    train_idx: np.ndarray,
    scaler: Any | None,
    chunk_size: int,
) -> np.ndarray:
    """Загружает train-данные чанками, применяя scaler, в один numpy-массив."""
    n_train = len(train_idx)
    n_features = X.shape[1]
    X_out = np.empty((n_train, n_features), dtype=np.float32)

    sorted_idx = np.sort(train_idx)
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = np.asarray(X[sorted_idx[start:end]], dtype=np.float32)
        if scaler is not None:
            chunk = scaler.transform(chunk)
        X_out[start:end] = chunk

    return X_out


# ============================================================
# 4. Цикл обучения (одна эпоха)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> float:
    """Обучает модель одну эпоху, возвращает средний loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


# ============================================================
# 5. Валидация
# ============================================================

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, list, list, list]:
    """Валидирует модель, возвращает (val_loss, preds, probs, true)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_probs, all_true = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())

    return total_loss / total_samples if total_samples > 0 else 0.0, all_preds, all_probs, all_true


# ============================================================
# 6. Оценка и логирование метрик
# ============================================================

def evaluate_and_log(
    all_true: list,
    all_preds: list,
    all_probs: list,
) -> None:
    """Логирует финальные метрики классификации."""
    logger.info("\n=== Final Metrics ===")
    logger.info("Accuracy:  %.4f", accuracy_score(all_true, all_preds))
    logger.info("Precision: %.4f", precision_score(all_true, all_preds, zero_division=0))
    logger.info("Recall:    %.4f", recall_score(all_true, all_preds, zero_division=0))
    logger.info("F1-score:  %.4f", f1_score(all_true, all_preds, zero_division=0))
    logger.info("ROC-AUC:   %.4f", roc_auc_score(all_true, all_probs))
    logger.info("LogLoss:   %.4f", log_loss(all_true, all_probs))
    logger.info("MCC:       %.4f", matthews_corrcoef(all_true, all_preds))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(all_true, all_preds))
    logger.info("\n%s", classification_report(all_true, all_preds, zero_division=0))


# ============================================================
# 7. Сохранение артефактов
# ============================================================

def save_artifacts(
    model: nn.Module,
    cfg: Config,
    n_features: int,
    feature_columns: list[str] | None = None,
    scaler: Any | None = None,
) -> None:
    """Сохраняет модель (с метаданными) и scaler (JSON)."""
    save_model(
        model,
        cfg.model.save_path,
        input_dim=n_features,
        cfg=cfg.model,
        feature_columns=feature_columns,
        n_features=n_features,
        extra={
            "optimizer": cfg.training.optimizer,
            "learning_rate": cfg.training.learning_rate,
            "split_strategy": cfg.training.split_strategy,
            "seed": cfg.training.seed,
            "scaler_type": cfg.training.scaler_type,
            "use_smote": cfg.training.use_smote,
            "pos_weight_mode": cfg.training.pos_weight_mode,
            "gradient_clip_norm": cfg.training.gradient_clip_norm,
            "scheduler": cfg.training.scheduler,
        },
    )

    if scaler is not None:
        scaler_path = Path(cfg.model.save_path).with_suffix(".scaler.json")
        save_scaler_json(scaler, scaler_path)


# ============================================================
# 8. Вычисление pos_weight для cost-sensitive learning (#17)
# ============================================================

def compute_pos_weight(cfg: Config, y_train: np.ndarray | None = None) -> float:
    """Вычисляет вес положительного класса для BCEWithLogitsLoss.

    Три режима (cfg.training.pos_weight_mode):
    - "statistical": из соотношения классов в train (negatives/positives).
      Это стабильно для обучения — градиент не «взрывается» при больших весах.
      Бизнес-стоимости учитываются позже, при оптимизации порога (#14).
    - "cost_matrix": из cost_matrix (cost_fn / cost_fp). Может быть очень
      большим (например 100) и дестабилизировать обучение.
    - "explicit": используется явно заданное cfg.training.pos_weight.
    """
    mode = cfg.training.pos_weight_mode

    if mode == "explicit" or cfg.training.pos_weight is not None:
        pos_weight = float(cfg.training.pos_weight) if cfg.training.pos_weight else 1.0
        logger.info("Explicit pos_weight=%.2f", pos_weight)
        return pos_weight

    if mode == "cost_matrix":
        ratio = cfg.cost_matrix.cost_fn / max(cfg.cost_matrix.cost_fp, 1e-9)
        logger.info(
            "pos_weight=%.2f (cost_matrix: cost_fn=%.0f / cost_fp=%.0f)",
            ratio,
            cfg.cost_matrix.cost_fn,
            cfg.cost_matrix.cost_fp,
        )
        return float(ratio)

    if mode == "statistical":
        if y_train is None:
            logger.warning(
                "pos_weight_mode='statistical' but y_train not provided. "
                "Using pos_weight=1.0"
            )
            return 1.0
        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        if n_pos == 0:
            logger.warning("No positive samples in train. pos_weight=1.0")
            return 1.0
        ratio = n_neg / n_pos
        logger.info(
            "pos_weight=%.2f (statistical: neg=%d / pos=%d)",
            ratio,
            int(n_neg),
            int(n_pos),
        )
        return float(ratio)

    raise ValueError(f"Unknown pos_weight_mode: {mode!r}")


# ============================================================
# 9. Построение LR scheduler
# ============================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> Any | None:
    """Создаёт LR scheduler по конфигурации."""
    scheduler_name = cfg.training.scheduler
    if scheduler_name == "none":
        return None
    if scheduler_name == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            min_lr=cfg.training.scheduler_min_lr,
        )
        logger.info(
            "ReduceLROnPlateau scheduler created (factor=%.2f, patience=%d)",
            cfg.training.scheduler_factor,
            cfg.training.scheduler_patience,
        )
        return scheduler
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
        logger.info("CosineAnnealingLR scheduler created (T_max=%d)", cfg.training.epochs)
        return scheduler
    raise ValueError(f"Unknown scheduler: {scheduler_name!r}")


# ============================================================
# 10. Основная функция обучения
# ============================================================

def train(cfg_path: str) -> None:
    """Полный цикл обучения."""
    from .config import load_config

    cfg = load_config(cfg_path)

    tracker = MLflowTracker(cfg.mlflow, cfg)
    with tracker:
        _train_inner(cfg, tracker)


def _train_inner(cfg: Config, tracker: MLflowTracker) -> None:
    """Внутренний цикл обучения (вызывается внутри MLflow context)."""
    set_seed(cfg.training.seed)

    # --- Данные ---
    data_info = prepare_data(cfg)
    n_features = data_info["n_features"]
    feature_columns = data_info["feature_columns"]
    x_path = data_info["x_path"]
    y_path = data_info["y_path"]

    X, y = load_memmap(x_path, y_path, n_features)
    logger.info("Loaded memmap: X=%s, y=%s", X.shape, y.shape)

    # --- Разбиение ---
    train_idx, val_idx = split_train_val(cfg, data_info)

    # --- Scaler (инкрементальный fit на train) ---
    scaler = fit_scaler(
        X,
        train_idx,
        scaler_type=cfg.training.scaler_type,
        chunk_size=cfg.data.chunk_size,
    )

    # --- y_train нужен для statistical pos_weight ---
    y_train_np = np.asarray(y[train_idx], dtype=np.float32)

    # --- Cost-sensitive learning ---
    pos_weight = compute_pos_weight(cfg, y_train=y_train_np)

    # --- DataLoader для val (поверх memmap, ленивый scaler) ---
    val_ds = MemmapDataset(X, y, val_idx, scaler=scaler)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and torch.cuda.is_available(),
    )

    # --- Train DataLoader ---
    if cfg.training.use_smote:
        # SMOTE требует загрузки train в ОЗУ
        X_train_np, y_train_np = apply_smote(X, y, train_idx, cfg, scaler=scaler)
        from torch.utils.data import TensorDataset

        train_ds = TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32),
        )
        del X_train_np
    elif cfg.training.precompute_train:
        # Предвычисление train-данных в память (с применённым scaler)
        # это УСКОРЯЕТ обучение в ~10-50x, т.к. scaler.transform применяется
        # один раз батчево, а не при каждой выборке в DataLoader.
        logger.info("Precomputing train data into memory (scaler applied once)...")
        X_train_np = _load_train_chunked(X, train_idx, scaler, cfg.data.chunk_size)
        from torch.utils.data import TensorDataset

        train_ds = TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32),
        )
        del X_train_np
    else:
        # Ленивая обработка через memmap (медленнее)
        train_ds = MemmapDataset(X, y, train_idx, scaler=scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and torch.cuda.is_available(),
        # persistent_workers работает только при num_workers > 0
        persistent_workers=cfg.training.num_workers > 0,
    )

    # --- Модель ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = build_model(n_features, cfg.model).to(device)
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = build_optimizer(model, cfg)

    # --- LR scheduler ---
    scheduler = build_scheduler(optimizer, cfg)

    # --- Цикл обучения ---
    best_loss = float("inf")
    patience_counter = 0
    all_preds: list = []
    all_probs: list = []
    all_true: list = []

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm=cfg.training.gradient_clip_norm,
        )

        val_loss, all_preds, all_probs, all_true = validate(
            model, val_loader, criterion, device, cfg.prediction.threshold
        )

        # Обновление LR scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f auc=%.4f lr=%.2e",
            epoch + 1,
            cfg.training.epochs,
            train_loss,
            val_loss,
            accuracy_score(all_true, all_preds),
            f1_score(all_true, all_preds, zero_division=0),
            roc_auc_score(all_true, all_probs),
            optimizer.param_groups[0]["lr"],
        )

        tracker.log_metrics(
            {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(accuracy_score(all_true, all_preds)),
                "val_f1": float(f1_score(all_true, all_preds, zero_division=0)),
                "val_roc_auc": float(roc_auc_score(all_true, all_probs)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            step=epoch,
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_artifacts(model, cfg, n_features, feature_columns, scaler)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # --- Финальные метрики ---
    evaluate_and_log(all_true, all_preds, all_probs)

    # --- Оптимизация порога (#14) ---
    all_true_arr = np.asarray(all_true)
    all_probs_arr = np.asarray(all_probs)
    curves_path = Path(cfg.model.save_path).parent / "roc_pr_curves.png"
    threshold_result = optimize_threshold_pipeline(
        y_true_val=all_true_arr,
        y_proba_val=all_probs_arr,
        cfg=cfg.threshold,
        cost_matrix=cfg.cost_matrix,
        curves_output_path=curves_path,
    )
    optimal_threshold = threshold_result["optimal_threshold"]
    logger.info("Optimal threshold: %.4f", optimal_threshold)
    tracker.log_metrics(threshold_result["metrics"])
    if "curves_path" in threshold_result:
        tracker.log_artifact(curves_path)

    # --- Сохранение конфига рядом с моделью (#15) ---
    config_path = Path(cfg.model.save_path).with_suffix(".config.json")
    save_config_artifact(cfg, config_path)
    tracker.log_artifact(config_path)


def main():
    parser = argparse.ArgumentParser(description="Train emergency predictor")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()