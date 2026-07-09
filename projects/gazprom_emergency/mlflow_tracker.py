"""MLflow трекинг экспериментов (замечание senior review #15).

Отслеживание экспериментов, версионирование моделей и данных:
- логирование метрик, параметров, артефактов,
- сохранение конфигурации рядом с чекпоинтом,
- регистрация моделей в Model Registry,
- stage-окружения (Staging/Production/Archived).

Использование::

    with MLflowTracker(cfg.mlflow, cfg) as tracker:
        tracker.log_params(...)
        tracker.log_metrics(...)
        # ... обучение ...
        tracker.log_artifact(model_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import Config, MlflowConfig, config_to_dict

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Контекстный менеджер для MLflow трекинга.

    Автоматически:
    - создаёт/подключается к эксперименту,
    - начинает/заканчивает run,
    - логирует параметры, метрики, артефакты.

    Если MLflow не установлен — работает как no-op (логирует warning).
    """

    def __init__(self, mlflow_cfg: MlflowConfig, full_cfg: Config) -> None:
        self.mlflow_cfg = mlflow_cfg
        self.full_cfg = full_cfg
        self._mlflow: Any = None
        self._run: Any = None
        self._active = False

    def __enter__(self) -> "MLflowTracker":
        if not self.mlflow_cfg.enabled:
            logger.info("MLflow tracking disabled (mlflow.enabled=False)")
            return self

        try:
            import mlflow

            self._mlflow = mlflow

            # Настройка tracking URI
            if self.mlflow_cfg.tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_cfg.tracking_uri)

            # Создаём эксперимент (если не существует)
            mlflow.set_experiment(self.mlflow_cfg.experiment_name)

            # Начинаем run
            self._run = mlflow.start_run()
            self._active = True

            # Логируем полный конфиг как параметры и артефакт
            self._log_config()

            logger.info(
                "MLflow run started: experiment=%s, run_id=%s",
                self.mlflow_cfg.experiment_name,
                self._run.info.run_id,
            )
        except ImportError:
            logger.warning(
                "MLflow is not installed. Install with: pip install mlflow. "
                "Tracking will be skipped."
            )
        except Exception as e:
            logger.warning("MLflow initialization failed: %s. Tracking skipped.", e)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._active and self._mlflow is not None:
            status = "FINISHED" if exc_type is None else "FAILED"
            self._mlflow.end_run(status=status)
            self._active = False
            logger.info("MLflow run ended (status=%s)", status)

    @property
    def is_active(self) -> bool:
        """Активен ли MLflow трекинг."""
        return self._active and self._mlflow is not None

    def log_params(self, params: dict[str, Any]) -> None:
        """Логирует параметры эксперимента."""
        if not self.is_active:
            return
        # MLflow не принимает вложенные структуры — сериализуем
        flat = _flatten_dict(params)
        self._mlflow.log_params(flat)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Логирует метрики эксперимента."""
        if not self.is_active:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Логирует файл-артефакт (модель, график, конфиг)."""
        if not self.is_active:
            return
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        """Логирует словарь как JSON-артефакт."""
        if not self.is_active:
            return
        self._mlflow.log_dict(data, artifact_file)

    def register_model(
        self,
        model_path: str | Path,
        model_name: str | None = None,
    ) -> str | None:
        """Регистрирует модель в Model Registry.

        Args:
            model_path: путь к модели (локальный или runs:/...).
            model_name: имя в реестре (из конфига, если None).

        Returns:
            Version модели в реестре (или None, если отключено).
        """
        if not self.is_active:
            return None

        registered_name = model_name or self.mlflow_cfg.registered_model_name
        if registered_name is None:
            logger.info("Model registration skipped (registered_model_name=None)")
            return None

        try:
            model_uri = f"runs:/{self._run.info.run_id}/{model_path}"
            mv = self._mlflow.register_model(model_uri, registered_name)
            logger.info(
                "Model registered: name=%s, version=%s",
                registered_name,
                mv.version,
            )

            # Transition to stage
            if self.mlflow_cfg.stage and self.mlflow_cfg.stage != "None":
                client = self._mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=registered_name,
                    version=mv.version,
                    stage=self.mlflow_cfg.stage,
                )
                logger.info(
                    "Model %s v%s transitioned to %s",
                    registered_name,
                    mv.version,
                    self.mlflow_cfg.stage,
                )

            return mv.version
        except Exception as e:
            logger.warning("Model registration failed: %s", e)
            return None

    def _log_config(self) -> None:
        """Логирует полную конфигурацию как параметры + JSON артефакт."""
        cfg_dict = config_to_dict(self.full_cfg)

        # Параметры (плоский dict)
        flat = _flatten_dict(cfg_dict)
        # MLflow лимит на длину значения параметра — 6000 символов
        truncated = {k: str(v)[:5000] for k, v in flat.items()}
        self._mlflow.log_params(truncated)

        # Полный конфиг как JSON-артефакт
        self._mlflow.log_dict(cfg_dict, "config.json")


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Преобразует вложенный dict в плоский (для MLflow params).

    Пример: {"a": {"b": 1}} → {"a.b": 1}
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, full_key))
        elif isinstance(value, list):
            result[full_key] = json.dumps(value)
        else:
            result[full_key] = value
    return result


def save_config_artifact(
    cfg: Config, output_path: str | Path
) -> Path:
    """Сохраняет конфигурацию как JSON рядом с моделью (замечание #15).

    Это позволяет узнать, на каких гиперпараметрах обучалась конкретная модель.

    Args:
        cfg: полная конфигурация.
        output_path: путь для сохранения (обычно рядом с .pth).

    Returns:
        Путь к сохранённому файлу.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = config_to_dict(cfg)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Config saved to %s", output_path)
    return output_path