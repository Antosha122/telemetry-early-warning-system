@echo off
REM ============================================================
REM  Запуск проекта Gazprom Emergency (Windows)
REM  Двойной клик по run.bat откроет интерактивное меню.
REM ============================================================

REM Переход в директорию скрипта
cd /d "%~dp0"

REM Попытка использовать venv, если он есть
if exist "Scripts\python.exe" (
    "Scripts\python.exe" run.py %*
) else (
    REM Иначе — системный Python
    python run.py %*
)

REM Пауза, чтобы окно не закрылось сразу при двойном клике
if "%1"=="" pause