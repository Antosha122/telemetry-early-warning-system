"""Оптимизация порога классификации и калибровка вероятностей (замечание #14).

Для дисбалансированной задачи аварий порог 0.5 почти наверняка неоптимален.

Поддерживаемые стратегии оптимизации порога:
- ``"f1"`` — максимизация F1-score,
- ``"recall_at_precision"`` — максимум recall при заданной минимальной precision,
- ``"cost"`` — минимизация бизнес-стоимости (cost_matrix),
- ``"youden"`` — индекс Юдена (TPR - FPR), максимизация separation.

Калибровка вероятностей:
- ``"isotonic"`` — изотоническая регрессия,
- ``"sigmoid"`` — Platt scaling (логистическая регрессия).

Также сохраняет ROC/PR-кривые как артефакты (PNG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CostMatrixConfig, ThresholdConfig

logger = logging.getLogger(__name__)


# ============================================================
# Поиск оптимального порога
# ============================================================


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cfg: ThresholdConfig,
    cost_matrix: CostMatrixConfig | None = None,
) -> float:
    """Находит оптимальный порог по выбранной метрике.

    Args:
        y_true: истинные метки (0/1).
        y_proba: предсказанные вероятности положительного класса.
        cfg: конфигурация оптимизации порога.
        cost_matrix: матрица стоимостей (для metric="cost").

    Returns:
        Оптимальный порог в диапазоне [0.0, 1.0].
    """
    metric = cfg.metric

    if metric == "f1":
        return _find_threshold_f1(y_true, y_proba)
    elif metric == "recall_at_precision":
        return _find_threshold_recall_at_precision(
            y_true, y_proba, cfg.min_precision
        )
    elif metric == "cost":
        if cost_matrix is None:
            cost_matrix = CostMatrixConfig()
        return _find_threshold_cost(y_true, y_proba, cost_matrix)
    elif metric == "youden":
        return _find_threshold_youden(y_true, y_proba)
    else:
        logger.warning("Unknown threshold metric %r, using 0.5", metric)
        return 0.5


def _find_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Максимизирует F1-score по всем возможным порогам."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # F1 = 2 * P * R / (P + R)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    # precision_recall_curve возвращает thresholds длиной n-1
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    logger.info(
        "Optimal threshold (F1): %.4f (F1=%.4f)",
        best_threshold,
        f1_scores[best_idx],
    )
    return best_threshold


def _find_threshold_recall_at_precision(
    y_true: np.ndarray, y_proba: np.ndarray, min_precision: float
) -> float:
    """Максимизирует recall при условии precision >= min_precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Находим пороги, где precision >= min_precision
    valid_mask = precision[:-1] >= min_precision
    if not valid_mask.any():
        logger.warning(
            "No threshold achieves precision >= %.2f, using 0.5", min_precision
        )
        return 0.5
    # Среди валидных выбираем максимальный recall
    valid_recall = recall[:-1][valid_mask]
    valid_thresholds = thresholds[valid_mask]
    best_idx = np.argmax(valid_recall)
    best_threshold = float(valid_thresholds[best_idx])
    logger.info(
        "Optimal threshold (recall@precision>=%.2f): %.4f (recall=%.4f, precision=%.4f)",
        min_precision,
        best_threshold,
        valid_recall[best_idx],
        precision[:-1][valid_mask][best_idx],
    )
    return best_threshold


def _find_threshold_cost(
    y_true: np.ndarray, y_proba: np.ndarray, cost_matrix: CostMatrixConfig
) -> float:
    """Минимизирует бизнес-стоимость (максимизирует выгоду)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float("-inf")
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        cost = cost_matrix.total_cost(tp, fp, fn, tn)
        if cost > best_cost:
            best_cost = cost
            best_threshold = float(t)

    logger.info(
        "Optimal threshold (cost): %.4f (cost=%.2f)", best_threshold, best_cost
    )
    return best_threshold


def _find_threshold_youden(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Максимизирует индекс Юдена (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold (Youden's J): %.4f (J=%.4f)",
        best_threshold,
        youden_j[best_idx],
    )
    return best_threshold


# ============================================================
# Калибровка вероятностей
# ============================================================


def calibrate_probabilities(
    y_train: np.ndarray,
    y_proba_train: np.ndarray,
    y_proba_val: np.ndarray,
    method: str = "isotonic",
) -> np.ndarray:
    """Калибрует вероятности на валидационной выборке.

    Калибровка важна, если вероятности используются в бизнес-логике
    (например, для приоритизации проверок).

    Args:
        y_train: истинные метки train.
        y_proba_train: предсказанные вероятности на train.
        y_proba_val: предсказанные вероятности на val (для калибровки).
        method: "isotonic" или "sigmoid" (Platt).

    Returns:
        Откалиброванные вероятности для val.
    """
    if method == "none":
        return y_proba_val

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    elif method == "sigmoid":
        calibrator = LogisticRegression()
    else:
        logger.warning("Unknown calibration method %r, skipping", method)
        return y_proba_val

    # Platt (sigmoid) требует 2D вход
    X_train = y_proba_train.reshape(-1, 1)
    X_val = y_proba_val.reshape(-1, 1)

    calibrator.fit(X_train, y_train)
    if method == "isotonic":
        calibrated = calibrator.predict(X_val)
    else:
        calibrated = calibrator.predict_proba(X_val)[:, 1]

    logger.info("Probabilities calibrated with %s", method)
    return np.asarray(calibrated, dtype=np.float64)


# ============================================================
# ROC/PR-кривые как артефакты
# ============================================================


def save_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    threshold: float | None = None,
) -> dict[str, float]:
    """Сохраняет ROC и PR-кривые как PNG и возвращает метрики.

    Args:
        y_true: истинные метки.
        y_proba: предсказанные вероятности.
        output_path: путь для сохранения PNG.
        threshold: опциональный порог для отметки на графике.

    Returns:
        Словарь с ROC-AUC и PR-AUC (average precision).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping ROC/PR curve PNG. "
            "Install with: pip install matplotlib"
        )
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
        return {"roc_auc": roc_auc, "pr_auc": pr_auc}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})", color="blue")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if threshold is not None:
        # Отмечаем точку порога
        idx = np.argmin(np.abs(roc_thresholds - threshold))
        axes[0].scatter([fpr[idx]], [tpr[idx]], color="red", s=100, zorder=5, label=f"threshold={threshold:.3f}")
        axes[0].legend()

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision, label=f"PR (AP={pr_auc:.4f})", color="green")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if threshold is not None and len(pr_thresholds) > 0:
        idx = np.argmin(np.abs(pr_thresholds - threshold))
        axes[1].scatter([recall[idx]], [precision[idx]], color="red", s=100, zorder=5, label=f"threshold={threshold:.3f}")
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC/PR curves saved to %s (ROC-AUC=%.4f, PR-AUC=%.4f)", output_path, roc_auc, pr_auc)
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


# ============================================================
# Полный пайплайн оптимизации порога
# ============================================================


def optimize_threshold_pipeline(
    y_true_val: np.ndarray,
    y_proba_val: np.ndarray,
    y_true_train: np.ndarray | None = None,
    y_proba_train: np.ndarray | None = None,
    cfg: ThresholdConfig | None = None,
    cost_matrix: CostMatrixConfig | None = None,
    curves_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Полный пайплайн: калибровка → оптимизация порога → метрики → артефакты.

    Args:
        y_true_val: истинные метки валидации.
        y_proba_val: предсказанные вероятности валидации.
        y_true_train: истинные метки train (для калибровки).
        y_proba_train: предсказанные вероятности train (для калибровки).
        cfg: конфигурация порога.
        cost_matrix: матрица стоимостей.
        curves_output_path: путь для сохранения кривых (None = не сохранять).

    Returns:
        Словарь с optimal_threshold, calibrated_proba (если калибровка), метриками.
    """
    cfg = cfg or ThresholdConfig()
    result: dict[str, Any] = {}

    # 1. Калибровка вероятностей
    calibrated_proba = y_proba_val
    if cfg.calibration != "none" and y_proba_train is not None and y_true_train is not None:
        calibrated_proba = calibrate_probabilities(
            y_true_train, y_proba_train, y_proba_val, cfg.calibration
        )
        result["calibrated"] = True
        result["calibration_method"] = cfg.calibration
    else:
        result["calibrated"] = False

    # 2. Поиск оптимального порога
    if cfg.optimize:
        threshold = find_optimal_threshold(
            y_true_val, calibrated_proba, cfg, cost_matrix
        )
    else:
        threshold = 0.5

    result["optimal_threshold"] = float(threshold)

    # 3. Метрики при оптимальном пороге
    preds = (calibrated_proba >= threshold).astype(int)
    result["metrics"] = {
        "f1": float(f1_score(y_true_val, preds, zero_division=0)),
        "precision": float(precision_score(y_true_val, preds, zero_division=0)),
        "recall": float(recall_score(y_true_val, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_val, calibrated_proba)),
        "pr_auc": float(average_precision_score(y_true_val, calibrated_proba)),
    }

    # 4. Сохранение ROC/PR-кривых
    if cfg.save_curves and curves_output_path is not None:
        curve_metrics = save_roc_pr_curves(
            y_true_val, calibrated_proba, curves_output_path, threshold
        )
        result["curves_path"] = str(curves_output_path)
        result["metrics"].update(curve_metrics)

    logger.info(
        "Threshold optimization complete: threshold=%.4f, f1=%.4f, roc_auc=%.4f",
        threshold,
        result["metrics"]["f1"],
        result["metrics"]["roc_auc"],
    )

    return result