# Gazprom ML — Прогнозирование аварий на объектах газоснабжения

<p align="center">
  <strong>Машинное обучение для предсказания аварийных ситуаций по телеметрии</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.1-red?logo=pytorch&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%2FPrototype-orange">
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
- [Пайплайн обучения](#-пайплайн-обучения)
- [Инференс (прогнозирование)](#-инференс-прогнозирование)
- [Архитектура модели](#-архитектура-модели)
- [Метрики качества](#-метрики-качества)
- [Артефакты](#-артефакты)
- [Архив экспериментов (legacy)](#-архив-экспериментов-legacy)
- [Документация](#-документация)
- [Воспроизводимость](#-воспроизводимость)
- [Устранение неисправностей](#-устранение-неисправностей)
- [Разработка](#-разработка)

---

## 🎯 О проекте

Проект предназначен для **прогнозирования аварийных ситуаций на объектах газоснабжения** (ГРП, ГРС и др.) на основе телеметрических данных. Модель анализирует 3600 признаков, собранных с датчиков, и предсказывает вероятность за 3 часа до возникновения аварии.

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
| **Тип модели** | Нейронная сеть (PyTorch MLP) |
| **Балансировка классов** | SMOTE (oversampling минорного класса) |
| **Обработка данных** | Потоковая (Polars LazyFrame + numpy memmap) |
| **Язык** | Python 3.10+ |
| **Фреймворк ML** | PyTorch ≥ 2.1 |
| **Лицензия** | Внутреннее использование |

---

## 📁 Структура репозитория

```
f:\Work\Gazprom\
│
├── projects/                              # Основной исходный код
│   └── gazprom_emergency/                 # Подпроект: прогноз аварий
│       ├── __init__.py                    # Версия пакета (0.1.0)
│       ├── __main__.py                    # CLI точка входа
│       ├── config.py                      # Загрузка и валидация YAML-конфигурации
│       ├── config.yaml                    # Параметры пайплайна
│       ├── data.py                        # Потоковая загрузка и merge данных
│       ├── model.py                       # Архитектура модели (PyTorch MLP)
│       ├── train.py                       # Обучение и валидация
│       ├── predict.py                     # Инференс (прогноз аварий)
│       └── requirements.txt               # Зависимости подпроекта
│
├── artifacts/                             # Большие файлы (git-ignored)
│   ├── datasets/                          # Исходные и обработанные датасеты
│   ├── models/                            # Сохранённые веса моделей (.pth)
│   ├── predictions/                       # Выходные предсказания
│   ├── logs/                              # TensorBoard events
│   └── tensorboard_runs/                  # Метрики по фолдам
│
├── Data/                                  # Справочные данные (небольшой объём)
│   ├── opers.csv                          # Журнал операций (~3 млн строк)
│   ├── stpa5000 (2).csv                   # Телеметрия (3600 признаков, ~5000 строк)
│   ├── params.xlsx                        # Справочник параметров
│   └── repairs.xlsx                       # Журнал ремонтов
│
├── legacy/                                # Архив экспериментальных скриптов (15 файлов)
│   ├── main_stpa_stochastic_optimizer.py
│   ├── main_stpa5000_stochastic_optimizer.py
│   ├── kfold_pca_smote_pytorch.py
│   ├── dask_stpa_pytorch.py
│   ├── dask_stpa_pytorch_incremental.py
│   ├── dask_stpa5000_pytorch.py
│   ├── chunked_stpa_pytorch.py
│   ├── numpy_memmap_stpa_pytorch.py
│   ├── pandas_stpa_metrics.py
│   ├── small_dataset_emerg2023_pytorch.py
│   ├── filter_fill_nan_stpa.py
│   ├── merge_opers_stpa_by_date.py
│   ├── downtime_analysis.py
│   ├── seasonality_analysis.py
│   └── inspect_stpa_dates.py
│
├── .gitignore
└── README.md
```

---

## 🔍 Описание задачи

### Постановка

По телеметрическим показаниям (3600 признаков `v_0`…`v_3599`, собранных в момент времени `batch_time`) предсказать, произойдёт ли авария в течение ближайших 3 часов.

**Целевая переменная:** `is_emergency` (0 — нет аварии, 1 — авария)

### Особенности данных

1. **Сильный дисбаланс классов** — аварийные ситуации составляют малую долю от всех наблюдений. Решается через SMOTE-oversampling.
2. **Высокая размерность** — 3600 признаков на одно наблюдение.
3. **Временной характер** — данные привязаны ко времени (`batch_time` / `date`), требуется корректный merge по дате.
4. **Пропуски и простои** — в данных присутствуют NaN-периоды (простои оборудования), требующие заполнения.

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

### `stpa5000 (2).csv` — телеметрия

| Колонка | Тип | Описание |
|---|---|---|
| `brigade_id` | int | ID бригады |
| `metric_id` | int | ID метрики |
| `batch_time` | datetime | Время измерения |
| `v_0` … `v_3599` | float | 3600 признаков телеметрии |

**Разделитель:** `;`
**Формат:** 3604 колонки

### Объединение

Данные объединяются по полю `batch_time` ↔ `date` (время измерения телеметрии сопоставляется с журналом операций).

---

## 🛠 Технический стек

| Компонент | Технология | Версия |
|---|---|---|
| ML-фреймворк | PyTorch | ≥ 2.1.0 |
| Обработка данных | Polars | ≥ 0.20.0 |
| Backend для I/O | PyArrow | ≥ 14.0.0 |
| Классика ML | scikit-learn | ≥ 1.3.0 |
| Балансировка | imbalanced-learn | ≥ 0.11.0 |
| Конфигурация | PyYAML | ≥ 6.0 |
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

### Обучение модели

```bash
python -m projects.gazprom_emergency train \
    --config projects/gazprom_emergency/config.yaml
```

### Прогнозирование

```bash
# По файлу телеметрии по умолчанию (из config.yaml)
python -m projects.gazprom_emergency predict \
    --config projects/gazprom_emergency/config.yaml

# По пользовательскому файлу
python -m projects.gazprom_emergency predict \
    --config projects/gazprom_emergency/config.yaml \
    --input path/to/new_telemetry.csv
```

---

## ⚙️ Конфигурация

Вся конфигурация пайплайна хранится в [`projects/gazprom_emergency/config.yaml`](projects/gazprom_emergency/config.yaml) и поддерживает подстановку переменных окружения в формате `${VAR}`.

### Параметры

#### Данные (`data`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `source_dir` | `${DATA_DIR}` | Каталог с исходными CSV |
| `opers_file` | `opers.csv` | Файл журнала операций |
| `stpa_file` | `stpa.csv` | Файл телеметрии |
| `processed_dir` | `${ARTIFACTS_DIR}/.../processed` | Каталог для memmap-файлов |
| `chunk_size` | `50000` | Размер чанка при потоковой загрузке |

#### Модель (`model`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `hidden_dims` | `[256, 128, 64, 32]` | Размеры скрытых слоёв MLP |
| `dropout` | `0.3` | Вероятность dropout |
| `save_path` | `${ARTIFACTS_DIR}/models/...pth` | Путь сохранения весов |

#### Обучение (`training`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `optimizer` | `adam` | Оптимизатор (`adam` / `rmsprop`) |
| `learning_rate` | `0.001` | Скорость обучения |
| `weight_decay` | `0.0001` | L2-регуляризация |
| `epochs` | `50` | Максимальное число эпох |
| `batch_size` | `256` | Размер батча |
| `use_smote` | `true` | Применять SMOTE |
| `smote_sampling_strategy` | `0.5` | Доля минорного класса после SMOTE |
| `test_size` | `0.2` | Доля тестовой выборки |
| `random_state` | `42` | Seed для воспроизводимости |
| `early_stopping_patience` | `7` | Patience для early stopping |

#### Предсказание (`prediction`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `threshold` | `0.5` | Порог бинарной классификации |
| `horizon_hours` | `3` | Горизонт прогноза (часы) |
| `output_path` | `${ARTIFACTS_DIR}/predictions/...csv` | Путь сохранения предсказаний |

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
│     ┌───────────────┐     ┌───────────────┐                         │
│     │  opers.csv    │     │  stpa.csv     │                         │
│     │ (is_emergency)│     │ (v_0..v_3599) │                         │
│     └───────┬───────┘     └───────┬───────┘                         │
│             │    Polars LazyFrame  │                                 │
│             └─────────┬───────────┘                                 │
│                       ▼                                              │
│              JOIN по batch_time                                      │
│                       │                                              │
│                       ▼                                              │
│              np.memmap (X_merged.npy, y_merged.npy)                  │
│              (стриминг чанками по 50 000 строк)                      │
│                                                                      │
│  3. ПРЕДОБРАБОТКА (train.py, dataset.py)                             │
│     ├── Хронологический split по batch_time (без data leakage)      │
│     ├── MinMaxScaler (инкрементальный fit на train, чанками)        │
│     └── SMOTE (oversampling минорного класса только на train)       │
│                                                                      │
│  4. ОБУЧЕНИЕ (train.py)                                              │
│     ├── DataLoader (batch_size=256, shuffle)                        │
│     ├── BCEWithLogitsLoss                                           │
│     ├── Adam (lr=0.001, weight_decay=0.0001)                       │
│     ├── Early Stopping (patience=7)                                 │
│     └── Сохранение лучшей модели по val_loss                         │
│                                                                      │
│  5. ОЦЕНКА                                                           │
│     ├── Accuracy, Precision, Recall, F1                              │
│     ├── ROC-AUC, LogLoss, MCC                                       │
│     └── Confusion Matrix + Classification Report                    │
│                                                                      │
│  6. СОХРАНЕНИЕ                                                        │
│     ├── model.pth (веса + метаданные архитектуры)                  │
│     └── model.scaler.json (MinMaxScaler, безопасный JSON)          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Потоковая обработка больших данных

Для работы с файлами, превышающими объём ОЗУ, используется двухпроходная стратегия:

1. **Первый проход** — Polars LazyFrame в стриминговом режиме подсчитывает количество строк.
2. **Создание memmap-файлов** нужного размера на диске.
3. **Второй проход** — данные пишутся чанками в memmap, обеспечивая O(1) доступ по индексу во время обучения.

---

## 🔮 Инференс (прогнозирование)

```bash
python -m projects.gazprom_emergency predict --config <path> [--input <csv>]
```

**Процесс:**

1. Загрузка обученной модели и scaler из `artifacts/models/`
2. Чтение входного CSV с телеметрией (`v_0`…`v_3599`)
3. Масштабирование признаков через сохранённый MinMaxScaler
4. Прямой проход через модель → `sigmoid(logits)` → вероятность
5. Бинарное решение по порогу (`threshold = 0.5`)
6. Сохранение результатов в CSV: `probability`, `prediction`, `threshold`

---

## 🧠 Архитектура модели

```
EmergencyPredictor (PyTorch MLP)
═══════════════════════════════════════════════════

Вход: 3600 признаков
  │
  ▼
┌──────────────────────────────────────────┐
│  Linear(3600, 256)                       │
│  BatchNorm1d(256)                        │
│  ReLU()                                  │  ← Блок 1
│  Dropout(0.3)                            │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(256, 128)                        │
│  BatchNorm1d(128)                        │
│  ReLU()                                  │  ← Блок 2
│  Dropout(0.3)                            │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(128, 64)                         │
│  BatchNorm1d(64)                         │
│  ReLU()                                  │  ← Блок 3
│  Dropout(0.3)                            │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(64, 32)                          │
│  BatchNorm1d(32)                         │
│  ReLU()                                  │  ← Блок 4
│  Dropout(0.3)                            │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│  Linear(32, 1)  →  logit                 │  ← Выход
└──────────────────────────────────────────┘

Loss: BCEWithLogitsLoss (численно стабильный)
Выход: sigmoid(logit) → P(авария в течение 3ч)
```

**Особенности архитектуры:**
- Каждый скрытый слой: `Linear → BatchNorm → ReLU → Dropout`
- BatchNorm стабилизирует обучение при высоких размерностях
- Dropout предотвращает переобучение
- BCEWithLogitsLoss объединяет Sigmoid + BCELoss для численной стабильности

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
| **LogLoss** | Логарифмическая потеря (качество вероятностей) |
| **MCC** | Matthews Correlation Coefficient (робастен к дисбалансу) |
| **Confusion Matrix** | Матрица ошибок (TP, FP, TN, FN) |

> ⚠️ При сильном дисбалансе классов приоритет отдаётся **Recall**, **F1** и **MCC**, а не Accuracy.

---

## 📦 Артефакты

Все большие файлы хранятся в `artifacts/` и **не коммитятся** в git (см. `.gitignore`).

### Структура

```
artifacts/
├── datasets/
│   └── gazprom_emergency/processed/
│       ├── X_merged.npy              # Матрица признаков (memmap)
│       └── y_merged.npy              # Вектор целевой переменной (memmap)
├── models/
│   ├── gazprom_emergency_model.pth   # Текущая модель (основной проект)
│   ├── gazprom_emergency_model.scaler.pkl  # MinMaxScaler
│   ├── best_model.pth                # Лучшая модель (legacy-эксперименты)
│   ├── best_model_fold_{1..5}.pth    # Модели по фолдам K-Fold CV
│   └── ...
├── predictions/
│   └── gazprom_emergency_predictions.csv
├── logs/                             # TensorBoard events
└── tensorboard_runs/                 # Структурированные runs по фолдам
```

### Просмотр TensorBoard

```bash
tensorboard --logdir artifacts/tensorboard_runs --port 6006
# Открыть http://localhost:6006 в браузере
```

### Версионирование больших файлов

Для версионирования артефактов рекомендуется [DVC](https://dvc.org/):

```bash
pip install dvc
dvc init
dvc add artifacts/models/gazprom_emergency_model.pth
```

---

## 📜 Архив экспериментов (legacy)

Папка `legacy/` содержит 15 экспериментальных скриптов, написанных в процессе исследования задачи. Они **не предназначены для production-использования** и сохранены для исторической справки.

| Скрипт | Подход | Описание |
|---|---|---|
| `main_stpa_stochastic_optimizer.py` | Custom optimizer | Кастомный `StochasticAdaptiveOptimizer` + MLP (6 слоёв по 256) |
| `main_stpa5000_stochastic_optimizer.py` | Custom optimizer | Вариант для stpa5000 датасета |
| `kfold_pca_smote_pytorch.py` | K-Fold + PCA | K-Fold CV (2 фолда), PCA (95% дисперсии), SMOTE + undersampling, TensorBoard |
| `dask_stpa_pytorch.py` | Dask | Распределённая обработка через Dask |
| `dask_stpa_pytorch_incremental.py` | Dask + Incremental | Инкрементальное обучение через `dask_ml.wrappers.Incremental` |
| `dask_stpa5000_pytorch.py` | Dask | Вариант для stpa5000 |
| `chunked_stpa_pytorch.py` | Chunked pandas | Чанковое чтение pandas (по 100 000 строк), MLP (5 слоёв по 512) |
| `numpy_memmap_stpa_pytorch.py` | numpy memmap | Прямой memmap над CSV (содержит ошибки) |
| `pandas_stpa_metrics.py` | Pandas | Полный пайплайн на pandas с метриками |
| `small_dataset_emerg2023_pytorch.py` | Small data | Упрощённая модель для малого датасета (emerg2023) |
| `filter_fill_nan_stpa.py` | Preprocessing | Очистка NaN и заполнение средними соседних строк |
| `merge_opers_stpa_by_date.py` | ETL | Слияние opers + stpa по дате (чанками) |
| `downtime_analysis.py` | EDA | Анализ периодов простоя (NaN-периодов) с визуализацией |
| `seasonality_analysis.py` | EDA | Анализ сезонности аварий и связей с ремонтами |
| `inspect_stpa_dates.py` | EDA | Инспекция дат в stpa-файле |

### Эволюция подходов

```
Pandas chunked (legacy)
       │
       ▼
Dask distributed (legacy)
       │
       ▼
numpy memmap (legacy)
       │
       ▼
Polars LazyFrame + memmap (projects/) ← текущая архитектура
```

---

## 📚 Документация

| Документ | Путь | Описание |
|---|---|---|
| Математическая модель | `docs/mathematical_model.docx` | Формальное описание модели и методов |
| Презентация | `docs/project_presentation.pptx` | Презентация проекта и результатов |

---

## 📌 Примечания

- **Разделитель CSV:** Файл `stpa5000 (2).csv` использует `;` как разделитель. Основной пайплайн (`data.py`) ожидает стандартный `,` — убедитесь, что данные приведены к единому формату.
- **GPU:** При наличии CUDA-совместимого GPU обучение автоматически переключается на GPU (`torch.cuda.is_available()`).
- **Воспроизводимость:** Глобальный `seed=42` фиксирует `torch`, `numpy`, `random` и `cudnn.deterministic` — воспроизводимость весов, разбиения и SMOTE.
- **Масштабируемость:** Архитектура на Polars + memmap позволяет обрабатывать файлы, превышающие объём ОЗУ.
- **Безопасность:** Модель загружается через `weights_only=True` (защита от RCE); scaler сериализуется в JSON (не pickle).

---

## 🎲 Воспроизводимость

### Детерминированность

Проект обеспечивает полную воспроизводимость результатов обучения через `set_seed()` в `utils.py`:

| Источник | Фиксация | Где |
|---|---|---|
| Разбиение train/val | `split_strategy: "chronological"` | `train.py` → `chronological_split_indices()` |
| SMOTE | `random_state=42` | `train.py` → `SMOTE(random_state=...)` |
| Конфигурация | `config.yaml` под версионным контролем | Все гиперпараметры в Git |
| Веса инициализации | `torch.manual_seed(42)` | `train.py` → `set_seed(cfg.training.seed)` |
| NumPy / random | `np.random.seed`, `random.seed` | `utils.py` → `set_seed()` |
| cuDNN | `cudnn.deterministic=True` | `utils.py` → `set_seed()` |

### Как это работает

При старте обучения вызывается `set_seed(cfg.training.seed)` (по умолчанию 42), который фиксирует все источники случайности:

```python
# utils.py — вызывается в начале train()
set_seed(42)
# → random.seed(42)
# → np.random.seed(42)
# → torch.manual_seed(42)
# → torch.cuda.manual_seed_all(42)
# → torch.backends.cudnn.deterministic = True
# → torch.backends.cudnn.benchmark = False
```

> ⚠️ Полная детерминированность на GPU не гарантируется из-за недетерминированных CUDA-операций. Для 100% детерминированности используйте CPU.

---

## 🛠 Устранение неисправностей

<details>
<summary><b>🔹 FileNotFoundError: opers file not found</b></summary>

**Причина:** Переменная окружения `DATA_DIR` не задана или указывает на несуществующий путь.

**Решение:** Проверьте значение переменной:

```bash
# PowerShell
echo $env:DATA_DIR

# Linux/Mac
echo $DATA_DIR
```

Убедитесь, что файлы `opers.csv` и `stpa.csv` существуют в указанной директории.
</details>

<details>
<summary><b>🔹 RuntimeError: CUDA out of memory</b></summary>

**Причина:** Недостаточно видеопамяти для обучения с текущим `batch_size`.

**Решение:** Уменьшите `batch_size` в `config.yaml`:

```yaml
training:
  batch_size: 64   # или 32
```

Альтернативно — используйте CPU, удалив/закомментировав строку `.to(device)`.
</details>

<details>
<summary><b>🔹 Polars: not enough memory</b></summary>

**Причина:** Polars требуется память для оптимизатора запросов.

**Решение:** Уменьшите `chunk_size` в `config.yaml`:

```yaml
data:
  chunk_size: 10000   # по умолчанию 50000
```
</details>

<details>
<summary><b>🔹 SMOTE: ValueError (недостаточно образцов минорного класса)</b></summary>

**Причина:** Слишком мало примеров положительного класса для синтеза.

**Решение:** Снизьте `smote_sampling_strategy` (например, до `0.3`) или увеличьте объём данных.
</details>

<details>
<summary><b>🔹 ValueError: Found input variables with inconsistent numbers of samples</b></summary>

**Причина:** Несоответствие размерностей `X` и `y` после merge.

**Решение:** Проверьте, что ключи `batch_time` / `date` совпадают по формату и типу в обоих файлах. Убедитесь, что разделитель CSV корректный (`,` для основного пайплайна).
</details>

---

## 👨‍💻 Разработка

### Запуск из исходников

```bash
# Установить в режиме разработки (editable)
pip install -e .

# Или запустить как модуль напрямую
python -m projects.gazprom_emergency --help
```

### Структура модуля

```python
projects.gazprom_emergency
├── config.py    → load_config(path) → Config (dataclass)
├── data.py      → merge_to_memmap(cfg), load_memmap(...), load_batch_times(...)
├── dataset.py   → MemmapDataset, chronological_split_indices()
├── model.py     → EmergencyPredictor, build_model(), save/load_model() (weights_only=True)
├── train.py     → train(cfg_path) — модульный цикл обучения (9 функций)
├── predict.py   → predict_batch(cfg_path, input_csv) — безопасный инференс
├── utils.py     → set_seed(), save/load_scaler_json(), fit_minmax_incremental()
└── __main__.py  → CLI: train | predict
```

### Добавление нового оптимизатора

1. Расширьте блок выбора оптимизатора в `train.py`:

```python
elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=0.9,
    )
```

2. Добавьте имя в конфигурацию:

```yaml
training:
  optimizer: "sgd"
```

### Рекомендации по расширению

- **Новые признаки:** Добавляйте колонки с префиксом `v_` — они автоматически подхватятся `_feature_columns()`.
- **Новые модели:** Реализуйте класс, наследуемый от `nn.Module`, и добавьте фабричную функцию в `model.py`.
- **Логирование:** Используйте `logging.getLogger(__name__)` для согласованности с существующим кодом.
- **Тестирование:** Перед коммитом убедитесь, что пайплайн запускается на небольшом сэмпле данных.


<p align="center">
  <sub>© 2024 Gazprom ML. Прогнозирование аварий на объектах газоснабжения.</sub>
</p>
