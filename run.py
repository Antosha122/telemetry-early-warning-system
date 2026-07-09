#!/usr/bin/env python3
"""Единый интерактивный лаунчер для проекта Gazprom Emergency.

Запуск:
    python run.py            # интерактивное меню
    python run.py train      # обучение напрямую
    python run.py predict    # предсказание напрямую
    python run.py --help     # справка

Все пути уже прописаны в config.yaml по умолчанию:
- Данные:     Data/ (opers.csv, stpa5000 (2).csv)
- Артефакты:  artifacts/ (модели, датасеты, предсказания)

Ничего настраивать не нужно — просто запустите.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Цвета для вывода в консоль (Windows/Linux совместимо)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEFAULT_CONFIG = "projects/gazprom_emergency/config.yaml"
REQUIRED_PACKAGES = ["torch", "numpy", "pandas", "polars", "sklearn", "yaml", "tqdm"]
REQUIRED_PIP_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "polars",
    "scikit-learn",
    "pyyaml",
    "tqdm",
]


def _enable_colors() -> None:
    """Включает поддержку ANSI-цветов в Windows CMD."""
    if sys.platform == "win32":
        os.system("")  # активируем VT100 в Windows 10+


def _color(text: str, color: str) -> str:
    """Оборачивает текст в цветной ANSI-код."""
    return f"{color}{text}{RESET}"


def _resolve_config(config_arg: str | None) -> str:
    """Возвращает путь к конфигу (аргумент или значение по умолчанию)."""
    if config_arg:
        return config_arg
    return DEFAULT_CONFIG


def _check_dependencies() -> bool:
    """Проверяет, установлены ли необходимые пакеты. Возвращает True если всё ОК."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(_color(f"\n✗ Отсутствуют зависимости: {', '.join(missing)}", RED))
        print(_color("\n  Установите все зависимости командой:", YELLOW))
        print(f"    pip install {' '.join(REQUIRED_PIP_PACKAGES)}")
        print(_color("  или:", YELLOW))
        print("    pip install -r projects/gazprom_emergency/requirements.txt")
        print()
        answer = input("  Установить сейчас? (y/N): ").strip().lower()
        if answer == "y":
            cmd = [sys.executable, "-m", "pip", "install"] + REQUIRED_PIP_PACKAGES
            print(f"  Запуск: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, cwd=str(Path.cwd()))
            if result.returncode == 0:
                print(_color("\n✓ Зависимости установлены.", GREEN))
                return True
            else:
                print(_color(f"\n✗ Ошибка установки (код {result.returncode}).", RED))
                return False
        return False
    return True


# ============================================================
# Действия (actions)
# ============================================================


def action_train(config: str | None = None) -> None:
    """Обучение модели прогноза аварий."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== ОБУЧЕНИЕ МОДЕЛИ ===", BOLD + CYAN))
    print(f"  Конфиг: {cfg_path}\n")

    if not _check_dependencies():
        return

    from projects.gazprom_emergency.train import train

    train(cfg_path)
    print(_color("\n✓ Обучение завершено.", GREEN))


def action_predict(config: str | None = None, input_csv: str | None = None) -> None:
    """Предсказание (инференс) обученной моделью."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== ПРЕДСКАЗАНИЕ ===", BOLD + CYAN))
    print(f"  Конфиг: {cfg_path}")
    if input_csv:
        print(f"  Входной CSV: {input_csv}")
    print()

    if not _check_dependencies():
        return

    from projects.gazprom_emergency.predict import predict_batch

    result = predict_batch(cfg_path, input_csv)
    print(_color(f"\n✓ Предсказания готовы. Строк: {len(result)}", GREEN))
    print("  Первые 5 строк:")
    print(result.head().to_string(index=False))


def action_prepare(config: str | None = None) -> None:
    """Подготовка датасета: объединение opers.csv + stpa.csv → memmap."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== ПОДГОТОВКА ДАННЫХ (merge → memmap) ===", BOLD + CYAN))
    print(f"  Конфиг: {cfg_path}\n")

    if not _check_dependencies():
        return

    from projects.gazprom_emergency.config import load_config
    from projects.gazprom_emergency.data import get_processed_paths, merge_to_memmap

    cfg = load_config(cfg_path)
    x_path, y_path, t_path = get_processed_paths(cfg.data)

    if x_path.exists():
        print(f"  {YELLOW}Внимание: memmap уже существует: {x_path}{RESET}")
        answer = input("  Пересоздать? (y/N): ").strip().lower()
        if answer != "y":
            print("  Отменено.")
            return

    x_path, y_path, n_features = merge_to_memmap(cfg.data)
    print(_color(f"\n✓ Данные подготовлены: X={x_path}, n_features={n_features}", GREEN))


def action_optimize(config: str | None = None) -> None:
    """Оптимизация порога (требуется обученная модель и валидационные данные)."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== ОПТИМИЗАЦИЯ ПОРОГА ===", BOLD + CYAN))
    print(f"  Конфиг: {cfg_path}\n")

    if not _check_dependencies():
        return

    print(
        _color(
            "  Эта команда запускает полный цикл обучения для расчёта "
            "оптимального порога.",
            YELLOW,
        )
    )
    print("  (Оптимизация порога встроена в обычное обучение.)\n")
    answer = input("  Запустить обучение с оптимизацией порога? (Y/n): ").strip().lower()
    if answer == "n":
        print("  Отменено.")
        return

    from projects.gazprom_emergency.train import train

    train(cfg_path)
    print(_color("\n✓ Оптимизация порога выполнена.", GREEN))


def action_compare(config: str | None = None) -> None:
    """Сравнение бейзлайн-моделей."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===", BOLD + CYAN))
    print(f"  Конфиг: {cfg_path}\n")

    if not _check_dependencies():
        return

    from projects.gazprom_emergency.baselines import available_baselines

    print("  Доступные модели для сравнения:")
    for name in available_baselines():
        print(f"    • {name}")

    from projects.gazprom_emergency.config import load_config

    cfg = load_config(cfg_path)
    models = cfg.experiment.models_to_compare
    print(f"\n  Модели из конфига: {models}")
    print(
        f"  Метрика выбора: {cfg.experiment.selection_metric}\n"
        f"  CV folds: {cfg.experiment.cv_folds}\n"
    )

    answer = input("  Запустить сравнение? (y/N): ").strip().lower()
    if answer != "y":
        print("  Отменено.")
        return

    _run_comparison(cfg)
    print(_color("\n✓ Сравнение завершено.", GREEN))


def _run_comparison(cfg) -> None:
    """Внутренняя логика сравнения моделей на подготовленных данных."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from projects.gazprom_emergency.baselines import compare_models, select_best_model
    from projects.gazprom_emergency.data import load_memmap
    from projects.gazprom_emergency.train import prepare_data
    from projects.gazprom_emergency.utils import fit_minmax_incremental

    data_info = prepare_data(cfg)
    x_path = data_info["x_path"]
    y_path = data_info["y_path"]
    n_features = data_info["n_features"]

    X, y = load_memmap(x_path, y_path, n_features)

    all_idx = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(
        all_idx, test_size=cfg.training.test_size, random_state=cfg.training.random_state
    )

    scaler = fit_minmax_incremental(
        X, train_idx, chunk_size=cfg.data.chunk_size, feature_range=(0.0, 1.0)
    )

    X_train = scaler.transform(np.asarray(X[train_idx], dtype=np.float32))
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    X_val = scaler.transform(np.asarray(X[val_idx], dtype=np.float32))
    y_val = np.asarray(y[val_idx], dtype=np.float32)

    baseline_models = [m for m in cfg.experiment.models_to_compare if m != "mlp"]
    if not baseline_models:
        print(_color("  Нет бейзлайн-моделей для сравнения.", YELLOW))
        return

    results = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        model_names=baseline_models,
        cfg=cfg,
        selection_metric=cfg.experiment.selection_metric,
    )

    best = select_best_model(results, cfg.experiment.selection_metric)
    print(_color(f"\n  Лучшая модель: {best}", GREEN))


def action_info(config: str | None = None) -> None:
    """Показывает информацию о конфигурации и артефактах."""
    cfg_path = _resolve_config(config)
    print(_color("\n=== ИНФОРМАЦИЯ О ПРОЕКТЕ ===", BOLD + CYAN))

    print(f"\n  {BOLD}Конфигурация ({cfg_path}):{RESET}")
    cfg = None
    try:
        from projects.gazprom_emergency.config import load_config

        cfg = load_config(cfg_path)
        print(f"    Данные:           {cfg.data.source_dir}")
        print(f"    opers_file:       {cfg.data.opers_file}")
        print(f"    stpa_file:        {cfg.data.stpa_file}")
        print(f"    processed_dir:    {cfg.data.processed_dir}")
        print(f"    Модель save_path: {cfg.model.save_path}")
        print(f"    Архитектура:      {cfg.model.hidden_dims} (dropout={cfg.model.dropout})")
        print(f"    Epochs:           {cfg.training.epochs}")
        print(f"    Batch size:       {cfg.training.batch_size}")
        print(f"    Use SMOTE:        {cfg.training.use_smote}")
        print(f"    Optimizer:        {cfg.training.optimizer} (lr={cfg.training.learning_rate})")
        print(f"    Threshold:        {cfg.prediction.threshold}")
        print(f"    Output:           {cfg.prediction.output_path}")
    except Exception as e:
        print(_color(f"    Ошибка загрузки конфига: {e}", RED))

    print(f"\n  {BOLD}Артефакты:{RESET}")
    artifacts_path = Path("artifacts")
    if artifacts_path.exists():
        model_path = Path(cfg.model.save_path) if cfg is not None else None
        if model_path and model_path.exists():
            print(f"    {GREEN}✓ Модель:{RESET} {model_path} ({model_path.stat().st_size} байт)")
        else:
            print(f"    {YELLOW}○ Модель не найдена (сначала обучите){RESET}")

        processed = Path(cfg.data.processed_dir) if cfg is not None else None
        if processed and processed.exists():
            files = list(processed.glob("*.npy"))
            print(f"    {GREEN}✓ Memmap-данные:{RESET} {processed} ({len(files)} файлов)")
        else:
            print(f"    {YELLOW}○ Memmap-данные не найдены (сначала подготовьте){RESET}")
    else:
        print(f"    {YELLOW}○ Каталог artifacts/ будет создан при первом запуске{RESET}")

    print(f"\n  {BOLD}Исходные данные:{RESET}")
    data_path = Path("Data")
    if data_path.exists():
        for f in data_path.iterdir():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {GREEN}✓{RESET} {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"    {RED}✗ Каталог Data/ не найден{RESET}")

    print()


def action_test(config: str | None = None) -> None:
    """Запуск pytest."""
    print(_color("\n=== ЗАПУСК ТЕСТОВ ===", BOLD + CYAN))
    print("  Запуск: pytest\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=str(Path.cwd()),
        )
        if result.returncode == 0:
            print(_color("\n✓ Все тесты пройдены.", GREEN))
        else:
            print(_color(f"\n✗ Тесты завершились с кодом {result.returncode}", RED))
    except FileNotFoundError:
        print(_color("\n✗ pytest не установлен. Установите: pip install pytest", RED))


# ============================================================
# Интерактивное меню
# ============================================================

MENU_ITEMS: list[tuple[str, str, Callable[..., None] | None]] = [
    ("1", "Обучить модель", action_train),
    ("2", "Сделать предсказание", action_predict),
    ("3", "Подготовить датасет (merge CSV → memmap)", action_prepare),
    ("4", "Оптимизация порога", action_optimize),
    ("5", "Сравнить модели", action_compare),
    ("6", "Информация о проекте", action_info),
    ("7", "Запустить тесты (pytest)", action_test),
    ("0", "Выход", None),
]


def _print_banner() -> None:
    """Печатает баннер программы."""
    banner = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    ГАЗПРОМ — Система прогнозирования аварий                      ║
║    ML-пайплайн на основе телеметрии (PyTorch MLP)                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(_color(banner, CYAN))


def _print_menu() -> None:
    """Печатает интерактивное меню."""
    print(_color("\n┌── ГЛАВНОЕ МЕНЮ ─────────────────────────────────────────┐", BOLD))
    for key, label, _ in MENU_ITEMS:
        print(f"│  {BOLD}[{key}]{RESET}  {label:<52}│")
    print(_color("└─────────────────────────────────────────────────────────┘\n", BOLD))


def _prompt_input(prompt: str) -> str:
    """Безопасный ввод с обработкой Ctrl+C."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(_color("\nВыход.", YELLOW))
        sys.exit(0)


def interactive_loop() -> None:
    """Главный интерактивный цикл."""
    _enable_colors()
    _print_banner()

    while True:
        _print_menu()
        choice = _prompt_input(_color("Выберите действие: ", BOLD))

        if choice == "0":
            print(_color("\nДо свидания! 👋\n", CYAN))
            break

        action = None
        for key, _, func in MENU_ITEMS:
            if key == choice:
                action = func
                break

        if action is None:
            print(_color("\n✗ Неверный выбор. Попробуйте снова.", RED))
            continue

        try:
            action()
        except KeyboardInterrupt:
            print(_color("\n\nОперация прервана пользователем.", YELLOW))
        except Exception as e:
            print(_color(f"\n✗ Ошибка: {e}", RED))
            import traceback

            traceback.print_exc()

        if choice != "6":
            _prompt_input(_color("\nНажмите Enter для возврата в меню...", YELLOW))


# ============================================================
# CLI режим (аргументы командной строки)
# ============================================================


def cli_mode() -> None:
    """Режим командной строки: python run.py train"""
    _enable_colors()

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Единый лаунчер проекта Gazprom Emergency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run.py                      # интерактивное меню
  python run.py train                # обучение
  python run.py predict              # предсказание
  python run.py predict --input X.csv
  python run.py prepare              # подготовка датасета
  python run.py info                 # информация о проекте
  python run.py compare              # сравнение моделей
  python run.py test                 # запуск pytest
""",
    )

    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Обучить модель")
    p_train.add_argument("--config", default=None, help="Путь к config.yaml")

    p_predict = sub.add_parser("predict", help="Сделать предсказание")
    p_predict.add_argument("--config", default=None, help="Путь к config.yaml")
    p_predict.add_argument("--input", default=None, help="Входной CSV для предсказания")

    p_prepare = sub.add_parser("prepare", help="Подготовить датасет (merge → memmap)")
    p_prepare.add_argument("--config", default=None, help="Путь к config.yaml")

    p_optimize = sub.add_parser("optimize", help="Оптимизация порога")
    p_optimize.add_argument("--config", default=None, help="Путь к config.yaml")

    p_compare = sub.add_parser("compare", help="Сравнить модели")
    p_compare.add_argument("--config", default=None, help="Путь к config.yaml")

    p_info = sub.add_parser("info", help="Информация о проекте")
    p_info.add_argument("--config", default=None, help="Путь к config.yaml")

    sub.add_parser("test", help="Запустить pytest")

    args = parser.parse_args()

    if args.command is None:
        interactive_loop()
    elif args.command == "train":
        action_train(args.config)
    elif args.command == "predict":
        action_predict(args.config, args.input)
    elif args.command == "prepare":
        action_prepare(args.config)
    elif args.command == "optimize":
        action_optimize(args.config)
    elif args.command == "compare":
        action_compare(args.config)
    elif args.command == "info":
        action_info(args.config)
    elif args.command == "test":
        action_test(None)


if __name__ == "__main__":
    cli_mode()