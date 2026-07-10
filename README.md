# Gazprom ML — Прогнозирование аварий на объектах газоснабжения

<p align="center">
  <strong>Production-ready ML-пайплайн для предсказания аварийных ситуаций по телеметрии</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.1-red?logo=pytorch&logoColor=white">
  <img alt="Polars" src="https://img.shields.io/badge/Polars-%E2%89%A50.20-orange?logo=polars">
  <img alt="Status" src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-pytest-success">
  <img alt="License" src="https://img.shields.io/badge/License-Internal-lightgrey">
</p>

---

## 📋 Содержание

- [О проекте](#-о-проекте)
- [Характеристика проекта](#-характеристика-проекта)
- [Структура репозитория](#-структура-репозитория)
- [Описание задачи](#-описание-задачи)
- [Данные](#-данные)
- [Технический стек](#-технический-стек)
- [Быстрый старт](#-быстрый-старт)
- [Конфигурация](#-конфигурация)
- [Пайплайн обучения](#-пайплайн-обучения)
- [Инференс (прогнозирование)](#-инференс-прогнозирование)
- [Архитектура модели](#-архитектура-модели)
- [Валидация данных](#-валидация-данных)
- [Feature Engineering](#-feature-engineering)
- [Сравнение моделей](#-сравнение-моделей)
- [Оптимизация порога](#-оптимизация-порога)
- [Cost-sensitive Learning](#-cost-sensitive-learning)
- [Эксперименты и MLflow](#-эксперименты-и-mlflow)
- [Метрики качества](#-метрики-качества)
- [Артефакты](#-артефакты)
- [Воспроизводимость](#-воспроизводимость)
- [Тестирование и качество кода](#-тестирование-и-качество-кода)
- [Архив экспериментов (legacy)](#-архив-экспериментов-legacy)
- [Разработка](#-разработка)
- [Устранение неисправностей](#-устранение-неисправностей)

---

## 🎯 О проекте

Проект предназначен для **прогнозирования аварийных ситуаций на объектах газоснабжения** (ГРП, ГРС и др.) на основе телеметрических данных. Модель анализирует 3600 признаков, собранных с датчиков, и предсказывает вероятность аварии за 3 часа до её возникновения.

### Бизнес-задача

Своевременное прогнозирование аварий позволяет:
- 🔸 Предотвратить аварийные отключения газоснабжения
- 🔸 Снизить экономические потери от простоев
- 🔸 Оптимизировать график планово-предупредительных ремонтов
- 🔸 Повысить безопасность эксплуатации газораспределительных сетей

---

## 📊 Характеристика проекта

| Параметр | Значение |
|---|---|
| **Тип задачи** | Бинарная классификация (авария / нет аварии) |
| **Горизонт прогноза** | 3 часа |
| **Объём данных** | ~3 млн записей операций, ~5000+ записей телеметрии (образец) |
| **Число признаков** | 3600 (v_0 … v_3599) |
| **Основная модель** | PyTorch MLP (EmergencyPredictor) |
| **Бейзлайны** | LogReg, Random Forest, XGBoost, LightGBM |
| **Балансировка классов** | Cost-sensitive learning (pos_weight) / SMOTE (опционально) |
| **Разбиение данных** | Хронологическое (без data leakage) или случайное (настраиваемо) |
| **Масштабирование** | StandardScaler / MinMaxScaler (инкрементальный fit) |
| **Обработка данных** | Потоковая (Polars LazyFrame + numpy memmap) |
| **Эксперимент-трекинг** | MLflow (опционально) |
| **Воспроизводимость** | Фиксация seed (torch, numpy, random, cudnn) |
| **Тестирование** | pytest + coverage |
| **Линтеры** | ruff, mypy, pre-commit hooks |
| **Язык** | Python 3.10+ |

---

## 📁 Структура репозитория

```
f:\Work\Gazprom\
│
├── projects/                              # Основной исходный код
│   └── gazprom_emergency/                 # Подпроект: прогноз аварий
│       ├── __init__.py                    # Версия пакета (0.1.0)
│       ├── __main__.py                    # CLI точка входа (train/predict)
│       ├── config.py                      # Типизированная конфигурация (dataclasses)
│       ├── config.yaml                    # Параметры пайплайна
│       ├── data.py                        # Потоковая загрузка и merge данных → memmap
│       ├── dataset.py                     # MemmapDataset + хронологическое разбиение
│       ├── model.py                       # Архитектура MLP + Registry Pattern
│       ├── train.py                       # Модульный цикл обучения (10 функций)
│       ├── predict.py                     # Безопасный инференс (weights_only=True)
│       ├── utils.py                       # set_seed, JSON-сериализация scaler, инкрементальный fit
│       ├── contracts.py                   # Protocols + ModelRegistry/OptimizerRegistry
│       ├── optimizers.py                  # Фабрика оптимизаторов (adam, adamw, sgd, rmsprop)
│       ├── features.py                    # Feature engineering (время, лаги, rolling, PCA)
│       ├── validation.py                  # Валидация входных данных (NaN/inf/дубликаты)
│       ├── baselines.py                   # Бейзлайны: LogReg, RF, XGBoost, LightGBM
│       ├── threshold.py                   # Оптимизация порога + калибровка вероятностей
│       ├── pipeline.py                    # sklearn Pipeline (защита от data leakage)
│       ├── mlflow_tracker.py              # MLflow трекинг экспериментов
│       └── requirements.txt               # Зависимости подпроекта
│
├── tests/                                 # Тесты (pytest)
│   ├── conftest.py                        # Общие fixtures
│   ├── test_config.py                     # Тесты парсинга конфига + env-var
│   ├── test_data.py                       # Тесты merge → memmap
│   ├── test_dataset.py                    # Тесты хронологического split
│   ├── test_model.py                      # Тесты модели и Registry
│   ├── test_predict.py                    # Тесты инференса
│   ├── test_utils.py                      # Тесты scaler сериализации
│   ├── test_contracts.py                  # Тесты Protocols/Registries
│   ├── test_feature_columns.py            # Тесты feature_columns JSON
│   ├── test_ml_practices.py               # Тесты validation/features/baselines/threshold/pipeline
│   └── __init__.py
│
├── artifacts/                             # Большие файлы (git-ignored)
│   ├── datasets/                          # Memmap-данные (X_merged.npy, y_merged.npy, t_merged.npy)
│   ├── models/                            # Сохранённые модели (.pth) + scaler (.json) + config (.json)
│   ├── predictions/                       # Выходные CSV с прогнозами
│   ├── logs/                              # ROC/PR-кривые, метрики
│   └── tensorboard_runs/                  # Метрики по фолдам
│
├── Data/                                  # Справочные данные (небольшой объём)
│
├── legacy/                                # Архив экспериментальных скриптов
│
├── run.py                                 # Единый лаунчер (интерактивное меню + CLI)
├── run.bat                                # Windows-лаунчер (двойной клик)
├── pyproject.toml                         # Упаковка пакета + ruff + mypy + pytest
├── .pre-commit-config.yaml                # pre-commit hooks (ruff, mypy)
└── README.md
```

---

## 🔍 Описание задачи

### Постановка

По телеметрическим показаниям (3600 признаков `v_0`…`v_3599`, собранных в момент времени `batch_time`) предсказать, произойдёт ли авария в течение ближайших 3 часов.

**Целевая переменная:** `is_emergency` (0 — нет аварии, 1 — авария)

### Особенности данных

1. **Сильный дисбаланс классов** — решается через cost-sensitive learning (`pos_weight`) или SMOTE (опционально).
2. **Высокая размерность** — 3600 признаков на наблюдение (опционально PCA).
3. **Временной характер** — данные привязаны ко времени (`batch_time` / `date`); используется хронологическое разбиение для предотвращения data leakage.
4. **Пропуски и простои** — валидация и обработка NaN/inf через `validation.py`.

### Гранулярность прогноза

Модель предсказывает аварийность **на уровне часа** (`batch_time`), а не отдельной операции. При подготовке данных метка агрегируется как `max(is_emergency)` — если хоть одна операция в час аварийная, весь час помечается аварийным. Опция `attach_opers_context: true` добавляет в выходной CSV колонки `n_ops`, `n_emergency_ops`, `ground_truth` для прозрачности.

---

## 📥 Данные

### `opers.csv` — журнал операций

| Колонка | Тип | Описание |
|---|---|---|
| `svod_opers_id` | int | ID сводной операции |
| `op_templ_id` | int | ID шаблона операции |
| `r_id` | int | ID объекта (repair) |
| `date` | datetime | Дата и время операции |
| `duration_hours` | int | Длительность (часы) |
| `is_emergency` | bool | **Целевая переменная**: авария (true/false) |

**Объём:** ~3 046 209 строк

### `stpa.csv` — телеметрия

| Колонка | Тип | Описание |
|---|---|---|
| `brigade_id` | int | ID бригады |
| `metric_id` | int | ID метрики |
| `batch_time` | datetime | Время измерения |
| `v_0` … `v_3599` | float | 3600 признаков телеметрии |

**Разделитель:** `;`
**Формат:** 3604 колонки (первая колонка может быть пустой — пропускается через `stpa_skip_first_column`)

### Объединение

Данные объединяются по полю `batch_time` ↔ `date` (время измерения телеметрии сопоставляется с журналом операций). Дубликаты `batch_time` в opers агрегируются как `max(is_emergency)`, в stpa — `unique(keep="first")`.

---

## 🛠 Технический стек

| Компонент | Технология | Версия |
|---|---|---|
| ML-фреймворк | PyTorch | ≥ 2.1.0 |
| Обработка данных | Polars | ≥ 0.20.0 |
| Backend для I/O | PyArrow | ≥ 14.0.0 |
| Классика ML | scikit-learn | ≥ 1.3.0 |
| Балансировка | imbalanced-learn | ≥ 0.11.0 |
| Бустинги (опц.) | XGBoost / LightGBM | опционально |
| Трекинг (опц.) | MLflow | опционально |
| Конфигурация | PyYAML | ≥ 6.0 |
| Тестирование | pytest + pytest-cov | ≥ 7.4.0 |
| Линтеры | ruff + mypy | ≥ 0.1.0 / ≥ 1.7.0 |
| Прогресс-бары | tqdm | ≥ 4.66.0 |
| Анализ данных | NumPy / Pandas | ≥ 1.26 / ≥ 2.1 |

---

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.10+
- pip
- (Опционально) CUDA-совместимый GPU для ускорения обучения

### Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd Gazprom

# 2. Создать виртуальное окружение
python -m venv venv

# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Установить зависимости
pip install -r projects/gazprom_emergency/requirements.txt

# или установить пакет в режиме разработки (с dev-инструментами):
pip install -e ".[dev]"
```

### Настройка путей к данным

```bash
# Linux/Mac
export DATA_DIR="/path/to/data"
export ARTIFACTS_DIR="/path/to/artifacts"

# Windows (PowerShell)
$env:DATA_DIR = "f:\Work\Gazprom\Data"
$env:ARTIFACTS_DIR = "f:\Work\Gazprom\artifacts"

# Windows (CMD)
set DATA_DIR=f:\Work\Gazprom\Data
set ARTIFACTS_DIR=f:\Work\Gazprom\artifacts
```

### Единый лаунчер (рекомендуемый способ)

Самый простой способ запуска — единый лаунчер `run.py` с интерактивным меню:

```bash
python run.py
```

При запуске без аргументов открывается меню с выбором действия:
1. Обучить модель
2. Сделать предсказание
3. Подготовить датасет (merge CSV → memmap)
4. Оптимизация порога
5. Сравнить модели (бейзлайны)
6. Информация о проекте
7. Запустить тесты

На Windows можно использовать `run.bat` (двойной клик).

Также доступен прямой запуск через CLI:

```bash
python run.py train                          # обучение
python run.py predict                        # предсказание
python run.py predict --input data.csv       # предсказание по конкретному файлу
python run.py prepare                        # подготовка датасета
python run.py compare                        # сравнение моделей
python run.py info                           # информация о проекте
python run.py test                           # запуск тестов
```

### Прямой запуск как Python-модуля

```bash
python -m projects.gazprom_emergency train --config projects/gazprom_emergency/config.yaml
python -m projects.gazprom_emergency predict --config projects/gazprom_emergency/config.yaml
```

После `pip install -e .` доступна консольная команда:

```bash
gazprom-emergency train --config projects/gazprom_emergency/config.yaml
```

---

## ⚙️ Конфигурация

Вся конфигурация пайплайна хранится в [`projects/gazprom_emergency/config.yaml`](projects/gazprom_emergency/config.yaml) и поддерживает подстановку переменных окружения в формате `${VAR}`. Конфигурация загружается в типизированные `dataclass`-объекты (`config.py`) с валидацией.

### Основные секции

#### Данные (`data`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `source_dir` | `Data` | Каталог с исходными CSV |
| `opers_file` | `opers.csv` | Файл журнала операций |
| `stpa_file` | `stpa5000 (2).csv` | Файл телеметрии |
| `processed_dir` | `artifacts/.../processed` | Каталог для memmap-файлов |
| `chunk_size` | `50000` | Размер чанка при потоковой загрузке |
| `opers_separator` / `stpa_separator` | `,` / `;` | Разделители CSV |
| `stpa_skip_first_column` | `true` | Пропускать пустую первую колонку |
| `opers_join_column` | `date` | Колонка времени в opers |

#### Модель (`model`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `architecture` | `emergency_predictor` | Имя архитектуры (через `ModelRegistry`) |
| `hidden_dims` | `[256, 128, 64, 32]` | Размеры скрытых слоёв MLP |
| `dropout` | `0.0` | Вероятность dropout |
| `save_path` | `artifacts/.../model.pth` | Путь сохранения весов |

#### Обучение (`training`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `optimizer` | `adam` | Оптимизатор (`adam`, `adamw`, `rmsprop`, `sgd`) |
| `learning_rate` | `0.003` | Скорость обучения |
| `weight_decay` | `0.0001` | L2-регуляризация |
| `epochs` | `30` | Максимальное число эпох |
| `batch_size` | `256` | Размер батча |
| `use_smote` | `false` | Применять SMOTE |
| `split_strategy` | `random` | `chronological` (без leakage) или `random` |
| `random_state` / `seed` | `42` | Seed для воспроизводимости |
| `scaler_type` | `standard` | `standard` (z-score), `minmax`, `none` |
| `pos_weight_mode` | `explicit` | `statistical`, `cost_matrix`, `explicit` |
| `pos_weight` | `0.15` | Вес положительного класса (для `explicit`) |
| `early_stopping_patience` | `10` | Patience для early stopping |
| `scheduler` | `reduce_on_plateau` | `none`, `reduce_on_plateau`, `cosine` |
| `gradient_clip_norm` | `null` | Норма градиента для clipping (null = выкл.) |

#### Предсказание (`prediction`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `threshold` | `0.78` | Статический порог (переопределяется оптимизированным из чекпоинта) |
| `horizon_hours` | `3` | Горизонт прогноза (часы) |
| `output_path` | `artifacts/.../predictions.csv` | Путь сохранения |
| `use_optimized_threshold` | `true` | Использовать порог из чекпоинта |
| `attach_opers_context` | `true` | Добавлять `n_ops`, `n_emergency_ops`, `ground_truth` |

#### Дополнительные секции

- **`validation`** — валидация входных данных (NaN/inf/дубликаты/типы).
- **`feature_engineering`** — временные признаки, лаги, rolling-статистики, PCA.
- **`experiment`** — сравнение моделей, K-Fold CV, Optuna.
- **`threshold`** — оптимизация порога (F1/Youden/cost) и калибровка.
- **`mlflow`** — трекинг экспериментов и регистрация моделей.
- **`cost_matrix`** — стоимости ошибок для cost-sensitive learning.

Полный список параметров с описаниями — в [`config.yaml`](projects/gazprom_emergency/config.yaml) и dataclass-определениях в [`config.py`](projects/gazprom_emergency/config.py).

---

## 🔄 Пайплайн обучения

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ОБУЧЕНИЕ МОДЕЛИ                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. ЗАГРУЗКА КОНФИГУРАЦИИ                                            │
│     config.yaml → load_config() → типизированный Config (dataclass) │
│                                                                      │
│  2. ПОДГОТОВКА ДАННЫХ (data.py)                                      │
│     ├── opers.csv + stpa.csv → Polars LazyFrame                      │
│     ├── Парсинг дат, каст типов, обработка дубликатов                │
│     ├── Дедупликация: opers → group_by(batch_time).agg(max)          │
│     ├── Дедупликация: stpa → unique(batch_time, keep="first")        │
│     ├── JOIN по batch_time (inner)                                   │
│     └── → np.memmap (X_merged.npy, y_merged.npy, t_merged.npy)       │
│                                                                      │
│  3. РАЗБИЕНИЕ (dataset.py)                                           │
│     ├── Хронологический split по batch_time (без data leakage)       │
│     └── или случайный split (настраиваемо)                           │
│                                                                      │
│  4. ПРЕДОБРАБОТКА (utils.py)                                         │
│     ├── Scaler: инкрементальный fit на train, чанками                │
│     │   (StandardScaler или MinMaxScaler)                            │
│     └── (опц.) SMOTE — только на train                               │
│                                                                      │
│  5. COST-SENSITIVE LEARNING                                          │
│     └── pos_weight для BCEWithLogitsLoss                             │
│         (statistical / cost_matrix / explicit)                       │
│                                                                      │
│  6. ОБУЧЕНИЕ (train.py)                                              │
│     ├── DataLoader (batch_size, num_workers, pin_memory)             │
│     ├── BCEWithLogitsLoss(pos_weight=...)                            │
│     ├── Adam (lr, weight_decay)                                      │
│     ├── Gradient clipping (опционально)                              │
│     ├── LR Scheduler: ReduceLROnPlateau / CosineAnnealingLR          │
│     ├── Early Stopping                                               │
│     └── Сохранение лучшей модели по val_loss                         │
│                                                                      │
│  7. ОЦЕНКА (evaluate_and_log)                                        │
│     ├── Accuracy, Precision, Recall, F1                              │
│     ├── ROC-AUC, PR-AUC, LogLoss, MCC                                │
│     └── Confusion Matrix + Classification Report                     │
│                                                                      │
│  8. ОПТИМИЗАЦИЯ ПОРОГА (threshold.py)                                │
│     ├── F1 / Youden / cost / recall@precision                        │
│     ├── Калибровка вероятностей (isotonic / sigmoid)                 │
│     └── ROC/PR-кривые (PNG артефакт)                                 │
│                                                                      │
│  9. СОХРАНЕНИЕ АРТЕФАКТОВ                                            │
│     ├── model.pth (веса + полные метаданные)                         │
│     ├── model.scaler.json (scaler параметры, без pickle)             │
│     └── model.config.json (полный конфиг обучения)                   │
│                                                                      │
│  10. MLFLOW ТРЕКИНГ (опционально)                                    │
│      ├── Логирование метрик/параметров/артефактов                    │
│      └── Регистрация модели в Model Registry                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Потоковая обработка больших данных

Для работы с файлами, превышающими объём ОЗУ, используется стратегия на базе memmap:

1. **Polars LazyFrame** — ленивое сканирование CSV с оптимизацией запросов.
2. **Создание memmap-файлов** нужного размера на диске.
3. **Чанковая запись** данных в memmap, обеспечивая O(1) доступ по индексу во время обучения.
4. **MemmapDataset** — PyTorch `Dataset` поверх `np.memmap`, читающий данные постранично через `__getitem__`.

---

## 🔮 Инференс (прогнозирование)

```bash
python run.py predict
# или
python run.py predict --input path/to/new_telemetry.csv
```

**Процесс:**

1. Загрузка метаданных чекпоинта (`feature_columns`, `input_dim`, `optimal_threshold`).
2. Приоритет порога: **оптимизированный из чекпоинта** > статический из конфига.
3. Загрузка модели (`weights_only=True` — защита от RCE) и scaler (JSON).
4. Чтение входного CSV с телеметрией (`v_0`…`v_3599`).
5. Масштабирование признаков через сохранённый scaler.
6. Прямой проход → `sigmoid(logits)` → вероятность.
7. Бинарное решение по порогу.
8. (Опц.) Присоединение контекста из `opers.csv` (`n_ops`, `n_emergency_ops`, `ground_truth`).
9. Сохранение результатов в CSV: `probability`, `prediction`, `threshold`, `batch_time`.

---

## 🧠 Архитектура модели

### EmergencyPredictor (PyTorch MLP)

```
Вход: 3600 признаков
  │
  ▼
┌──────────────────────────────────────────┐
│  Linear(3600, 256)                       │
│  BatchNorm1d(256)                        │
│  ReLU()                                  │  ← Блок 1
│  Dropout(p)                              │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(256, 128) + BN + ReLU + Dropout  │  ← Блок 2
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(128, 64) + BN + ReLU + Dropout   │  ← Блок 3
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(64, 32) + BN + ReLU + Dropout    │  ← Блок 4
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(32, 1)  →  logit                 │  ← Выход
└──────────────────────────────────────────┘

Loss: BCEWithLogitsLoss (численно стабильный, поддерживает pos_weight)
Выход: sigmoid(logit) → P(авария в течение 3ч)
```

### Расширяемость через Registry Pattern

Архитектуры и оптимизаторы регистрируются декларативно через `@ModelRegistry.register(...)` / `@OptimizerRegistry.register(...)`, что следует Open/Closed Principle — добавление новой модели/оптимизатора не требует правки ядра.

```python
# Добавление новой архитектуры
@ModelRegistry.register("transformer_encoder")
def _build_transformer(*, input_dim, model_cfg, **_):
    return MyTransformer(input_dim, ...)
```

### Безопасная сериализация

- **Веса модели**: `torch.save` с полными метаданными (`input_dim`, `hidden_dims`, `dropout`, `feature_columns`, `config_version`).
- **Загрузка**: `weights_only=True` (PyTorch ≥ 2.0) — защита от RCE.
- **Scaler**: JSON (не pickle) — параметры `StandardScaler`/`MinMaxScaler` тривиально сериализуемы.

---

## 🛡 Валидация данных

Модуль [`validation.py`](projects/gazprom_emergency/validation.py) проверяет и очищает входные данные перед merge:

| Проверка | Стратегия | Описание |
|---|---|---|
| **Дубликаты `batch_time`** | `fail` / `drop` / `aggregate` | Защита от many-to-many join |
| **NaN в фичах** | `fail` / `fill_mean` / `fill_median` / `fill_zero` / `drop` | Обработка пропусков |
| **Inf в фичах** | `fail` / `replace_with_nan` | Замена бесконечностей |
| **Целевая переменная** | `strict_binary_target` | Проверка ∈ {0, 1} |
| **Типы колонок** | `expected_feature_dtype` | Приведение к float32/float64 |
| **Кратность join** | `max_join_multiplicity` | Защита от размножения строк |
| **Доля NaN в строке** | `max_nan_fraction_per_row` | Отбрасывание строк с >90% NaN |

Формируется `ValidationReport` со статистикой и предупреждениями.

---

## 🧬 Feature Engineering

Модуль [`features.py`](projects/gazprom_emergency/features.py) (sklearn-совместимый трансформер `FeatureEngineer`):

| Тип признаков | Описание |
|---|---|
| **Временные** | hour/dayofweek/month (циклическое sin/cos кодирование), one-hot сезона (8+4 колонок) |
| **Лаги** | Значения признака `L` строк назад (lag_sizes: [1, 3, 6]) |
| **Rolling статистики** | Скользящие mean/std/min/max по окнам [3, 6, 12] (leakage-safe, исключая текущую строку) |
| **PCA** | Уменьшение размерности (float = дисперсия, int = число компонент) |
| **Обработка пропусков** | `interpolate_nan_rows()` — интерполяция соседних строк |

Все преобразования реализованы как sklearn-совместимые трансформеры и встраиваются в `sklearn.Pipeline` (через `pipeline.py`) для защиты от data leakage: `fit` только на train, `transform` на val/test.

---

## 📊 Сравнение моделей

Модуль [`baselines.py`](projects/gazprom_emergency/baselines.py) добавляет классические модели для табличных данных:

| Модель | Ключ реестра | Описание |
|---|---|---|
| Logistic Regression | `logreg` | Простейший линейный бейзлайн |
| Random Forest | `random_forest` | Ансамбль деревьев решений |
| XGBoost | `xgboost` | Градиентный бустинг (требует `pip install xgboost`) |
| LightGBM | `lightgbm` | Градиентный бустинг (требует `pip install lightgbm`) |

Доступные функции:
- `compare_models()` — сравнение на train/val по метрикам (PR-AUC, ROC-AUC, F1, cost).
- `select_best_model()` — выбор лучшей по заданной метрике.
- `cross_validate_baseline()` — K-Fold Cross-Validation (StratifiedKFold).

Конфигурация через секцию `experiment` (`models_to_compare`, `selection_metric`, `cv_folds`).

---

## 🎯 Оптимизация порога

Модуль [`threshold.py`](projects/gazprom_emergency/threshold.py) — для дисбалансированной задачи порог 0.5 почти наверняка неоптимален.

| Стратегия | Описание |
|---|---|
| `f1` | Максимизация F1-score |
| `recall_at_precision` | Максимальный recall при precision ≥ `min_precision` |
| `cost` | Минимизация бизнес-стоимости (через `cost_matrix`) |
| `youden` | Индекс Юдена (TPR − FPR) |

Дополнительно:
- **Калибровка вероятностей**: `isotonic` (изотоническая регрессия), `sigmoid` (Platt scaling).
- **ROC/PR-кривые**: сохраняются как PNG-артефакт с отмеченным оптимальным порогом.

Оптимизированный порог сохраняется в чекпоинт (`extra.optimal_threshold`) и автоматически используется при инференсе (если `prediction.use_optimized_threshold: true`).

---

## 💰 Cost-sensitive Learning

Модуль учитывает асимметрию стоимости ошибок (пропущенная авария дороже ложной тревоги):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `cost_fn` | `1000.0` | Стоимость пропущенной аварии (false negative) |
| `cost_fp` | `10.0` | Стоимость ложной тревоги (false positive) |
| `benefit_tp` | `990.0` | Выгода от предотвращённой аварии |
| `benefit_tn` | `0.0` | Выгода от корректного «нет аварии» |

Три режима `pos_weight` в `BCEWithLogitsLoss`:
- **`statistical`** — из соотношения классов в train (`negatives / positives`), стабильно.
- **`cost_matrix`** — из `cost_fn / cost_fp` (может дестабилизировать обучение).
- **`explicit`** — явно заданное значение `pos_weight`.

---

## 🧪 Эксперименты и MLflow

Модуль [`mlflow_tracker.py`](projects/gazprom_emergency/mlflow_tracker.py) (контекстный менеджер `MLflowTracker`):

- Логирование метрик, параметров, артефактов.
- Сохранение полного конфига как JSON рядом с чекпоинтом.
- Регистрация моделей в MLflow Model Registry с переходом на stage (Staging/Production).
- Если MLflow не установлен — работает как no-op (не ломает пайплайн).

Включение:

```yaml
mlflow:
  enabled: true
  tracking_uri: "http://localhost:5000"
  experiment_name: "gazprom_emergency"
  registered_model_name: "emergency_predictor"
  stage: "Staging"
```

---

## 📈 Метрики качества

Модель оценивается по следующим метрикам:

| Метрика | Описание |
|---|---|
| **Accuracy** | Общая доля правильных предсказаний |
| **Precision** | Доля истинных аварий среди предсказанных |
| **Recall** | Доля обнаруженных аварий из всех реальных |
| **F1-Score** | Гармоническое среднее Precision и Recall |
| **ROC-AUC** | Площадь под ROC-кривой (разделяющая способность) |
| **PR-AUC** | Average Precision (важно при дисбалансе) |
| **LogLoss** | Логарифмическая потеря (качество вероятностей) |
| **MCC** | Matthews Correlation Coefficient (робастен к дисбалансу) |
| **Confusion Matrix** | Матрица ошибок (TP, FP, TN, FN) |
| **Cost** | Бизнес-стоимость (через `cost_matrix`) |

> ⚠️ При сильном дисбалансе классов приоритет отдаётся **Recall**, **F1**, **PR-AUC** и **MCC**, а не Accuracy.

---

## 📦 Артефакты

Все большие файлы хранятся в `artifacts/` и **не коммитятся** в git (см. `.gitignore`).

```
artifacts/
├── datasets/
│   └── gazprom_emergency/processed/
│       ├── X_merged.npy              # Матрица признаков (memmap)
│       ├── y_merged.npy              # Вектор целевой переменной (memmap)
│       ├── t_merged.npy              # Временные метки batch_time (memmap)
│       └── feature_columns.json      # Список признаков (единый источник правды)
├── models/
│   ├── gazprom_emergency_model.pth   # Веса + метаданные архитектуры
│   ├── gazprom_emergency_model.scaler.json   # Scaler параметры (JSON)
│   ├── gazprom_emergency_model.config.json   # Полный конфиг обучения
│   └── roc_pr_curves.png             # ROC/PR-кривые с оптимальным порогом
├── predictions/
│   └── gazprom_emergency_predictions.csv
└── tensorboard_runs/                 # Метрики по фолдам (legacy)
```

### Просмотр TensorBoard

```bash
tensorboard --logdir artifacts/tensorboard_runs --port 6006
# Открыть http://localhost:6006 в браузере
```

---

## 🎲 Воспроизводимость

### Детерминированность

Проект обеспечивает воспроизводимость результатов обучения через `set_seed()` в `utils.py`:

| Источник | Фиксация | Где |
|---|---|---|
| Разбиение train/val | `split_strategy: "chronological"` | `train.py` → `chronological_split_indices()` |
| Конфигурация | `config.yaml` под версионным контролем | Все гиперпараметры в Git |
| Веса инициализации | `torch.manual_seed(seed)` | `train.py` → `set_seed()` |
| NumPy / random | `np.random.seed`, `random.seed` | `utils.py` → `set_seed()` |
| cuDNN | `cudnn.deterministic=True`, `benchmark=False` | `utils.py` → `set_seed()` |
| SMOTE | `random_state=42` | `train.py` → `SMOTE(random_state=...)` |
| Scaler | Инкрементальный fit (детерминированный) | `utils.py` → `fit_scaler()` |

> ⚠️ Полная детерминированность на GPU не гарантируется из-за недетерминированных CUDA-операций. Для 100% детерминированности используйте CPU.

---

## ✅ Тестирование и качество кода

### Тесты (pytest)

```bash
# Запуск всех тестов
python run.py test
# или
pytest tests/ -v

# С покрытием
pytest tests/ --cov=projects --cov-report=term-missing
```

Покрытие:
- `test_config.py` — парсинг YAML, подстановка env-var, edge-cases.
- `test_data.py` — merge → memmap, корректность данных, обработка ошибок.
- `test_dataset.py` — хронологическое разбиение.
- `test_model.py` — архитектура, Registry, save/load.
- `test_predict.py` — инференс, извлечение признаков.
- `test_utils.py` — сериализация scaler, инкрементальный fit.
- `test_contracts.py` — Protocols и Registries.
- `test_feature_columns.py` — JSON со списком признаков.
- `test_ml_practices.py` — validation, features, baselines, threshold, pipeline.

### Линтеры и форматирование

| Инструмент | Назначение | Конфигурация |
|---|---|---|
| **ruff** | Линтинг + форматирование | `pyproject.toml` → `[tool.ruff]` |
| **mypy** | Проверка типов | `pyproject.toml` → `[tool.mypy]` |
| **pre-commit** | Автоматизация перед коммитом | `.pre-commit-config.yaml` |

Установка pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

---

## 📜 Архив экспериментов (legacy)

Папка `legacy/` содержит 15 экспериментальных скриптов, написанных в процессе исследования задачи. Они **не предназначены для production-использования** и сохранены для исторической справки.

| Скрипт | Подход | Описание |
|---|---|---|
| `main_stpa_stochastic_optimizer.py` | Custom optimizer | Кастомный `StochasticAdaptiveOptimizer` + MLP |
| `main_stpa5000_stochastic_optimizer.py` | Custom optimizer | Вариант для stpa5000 датасета |
| `kfold_pca_smote_pytorch.py` | K-Fold + PCA | K-Fold CV, PCA (95%), SMOTE + undersampling |
| `dask_stpa_pytorch.py` | Dask | Распределённая обработка через Dask |
| `dask_stpa_pytorch_incremental.py` | Dask + Incremental | Инкрементальное обучение |
| `dask_stpa5000_pytorch.py` | Dask | Вариант для stpa5000 |
| `chunked_stpa_pytorch.py` | Chunked pandas | Чанковое чтение pandas |
| `numpy_memmap_stpa_pytorch.py` | numpy memmap | Прямой memmap над CSV |
| `pandas_stpa_metrics.py` | Pandas | Полный пайплайн на pandas |
| `small_dataset_emerg2023_pytorch.py` | Small data | Упрощённая модель для emerg2023 |
| `filter_fill_nan_stpa.py` | Preprocessing | Очистка NaN и интерполяция |
| `merge_opers_stpa_by_date.py` | ETL | Слияние opers + stpa по дате |
| `downtime_analysis.py` | EDA | Анализ периодов простоя |
| `seasonality_analysis.py` | EDA | Анализ сезонности аварий |
| `inspect_stpa_dates.py` | EDA | Инспекция дат в stpa-файле |

### Эволюция подходов

```
Pandas chunked (legacy) → Dask (legacy) → numpy memmap (legacy)
    → Polars LazyFrame + memmap (projects/) ← текущая архитектура
```

---

## 👨‍💻 Разработка

### Структура модуля

```python
projects.gazprom_emergency
├── config.py          → load_config(path) → Config (dataclass)
├── data.py            → merge_to_memmap(cfg), load_memmap(...), feature_columns
├── dataset.py         → MemmapDataset, chronological_split_indices()
├── model.py           → EmergencyPredictor, build_model(), save/load_model() (weights_only=True)
├── optimizers.py      → build_optimizer() (Registry: adam, adamw, sgd, rmsprop)
├── train.py           → train(cfg_path) — модульный цикл (10 функций)
├── predict.py         → predict_batch(cfg_path, input_csv) — безопасный инференс
├── utils.py           → set_seed(), save/load_scaler_json(), fit_scaler()
├── contracts.py       → Protocols + ModelRegistry + OptimizerRegistry
├── features.py        → FeatureEngineer (время, лаги, rolling, PCA)
├── validation.py      → validate_inputs() (NaN/inf/дубликаты/типы)
├── baselines.py       → LogReg, RF, XGBoost, LightGBM + compare_models()
├── threshold.py       → optimize_threshold_pipeline() + калибровка
├── pipeline.py        → sklearn Pipeline (leakage-safe)
├── mlflow_tracker.py  → MLflowTracker (контекстный менеджер)
└── __main__.py        → CLI: train | predict
```

### Добавление новой модели

1. Зарегистрируйте архитектуру через декоратор в `model.py`:

```python
@ModelRegistry.register("residual_mlp")
def _build_residual_mlp(*, input_dim: int, model_cfg: ModelConfig, **_) -> nn.Module:
    return MyResidualMLP(input_dim=input_dim, hidden_dims=model_cfg.hidden_dims)
```

2. Укажите имя в конфигурации:

```yaml
model:
  architecture: "residual_mlp"
```

### Добавление нового оптимизатора

```python
# optimizers.py
@OptimizerRegistry.register("adamw")
def _build_adamw(model, *, lr=0.001, weight_decay=0.01, **_):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

### Рекомендации по расширению

- **Новые признаки:** добавляйте колонки с префиксом `v_` — они автоматически подхватятся `_feature_columns()`.
- **Feature engineering:** используйте `FeatureEngineer` из `features.py` или `build_full_pipeline()` из `pipeline.py`.
- **Логирование:** используйте `logging.getLogger(__name__)` для согласованности.
- **Тестирование:** перед коммитом убедитесь, что `pytest tests/` проходит.

---

## 🛠 Устранение неисправностей

<details>
<summary><b>🔹 FileNotFoundError: opers file not found</b></summary>

**Причина:** Переменная окружения `DATA_DIR` не задана или указывает на несуществующий путь.

**Решение:** Проверьте значение переменной и наличие файлов `opers.csv` и `stpa.csv` в указанной директории.
</details>

<details>
<summary><b>🔹 RuntimeError: CUDA out of memory</b></summary>

**Причина:** Недостаточно видеопамяти для обучения с текущим `batch_size`.

**Решение:** Уменьшите `batch_size` в `config.yaml` (например, до 64 или 32) или используйте CPU.
</details>

<details>
<summary><b>🔹 Polars: not enough memory</b></summary>

**Причина:** Polars требуется память для оптимизатора запросов.

**Решение:** Уменьшите `chunk_size` в `config.yaml` (например, до 10000).
</details>

<details>
<summary><b>🔹 Dim mismatch: model=N, input=M</b></summary>

**Причина:** Несоответствие числа признаков между обучением и инференсом.

**Решение:** Убедитесь, что входной CSV содержит те же колонки `v_*`, что использовались при обучении. Список признаков сохранён в `feature_columns.json` и в метаданных чекпоинта. При необходимости перезапустите `python run.py prepare`.
</details>

<details>
<summary><b>🔹 SMOTE: ValueError (недостаточно образцов минорного класса)</b></summary>

**Причина:** Слишком мало примеров положительного класса для синтеза.

**Решение:** Снизьте `smote_sampling_strategy` (например, до `0.3`) или переключитесь на cost-sensitive learning (`use_smote: false`, `pos_weight_mode: "statistical"`).
</details>

<details>
<summary><b>🔹 ImportError: XGBoost/LightGBM/MLflow is not installed</b></summary>

**Решение:** Установите опциональные зависимости:

```bash
pip install xgboost lightgbm mlflow
```
</details>

---

## 📌 Примечания

- **Разделитель CSV:** Файл `stpa5000 (2).csv` использует `;` как разделитель (настроено в `stpa_separator`).
- **GPU:** При наличии CUDA-совместимого GPU обучение автоматически переключается на GPU.
- **Безопасность:** Модель загружается через `weights_only=True` (защита от RCE); scaler сериализуется в JSON (не pickle).
- **Масштабируемость:** Архитектура на Polars + memmap позволяет обрабатывать файлы, превышающие объём ОЗУ.
- **Единый источник правды:** Список признаков (`feature_columns.json`) используется и в обучении, и в инференсе — исключает рассинхрон.

---

<p align="center">
  <sub>© 2024 Gazprom ML. Прогнозирование аварий на объектах газоснабжения.</sub>
</p>