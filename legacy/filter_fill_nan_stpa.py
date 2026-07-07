import pandas as pd
import numpy as np
from tqdm import tqdm

print("Загрузка stpa")
# Путь к исходному файлу и путь для сохранения очищенного файла
input_file_path = 'F:/ККАЛ/DataSet Газпром/stpa.csv' # Замените на путь к вашему файлу
output_file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv' # Замените на желаемый путь для сохранения

# Размер чанка
chunksize = 10**5

# Определение диапазона столбцов v_0 до v_3599
v_columns = [f'v_{i}' for i in range(3600)]

# Подсчет общего количества строк для отображения прогресса
total_lines = sum(1 for _ in open(input_file_path)) - 1  # Минус 1 из-за заголовка

# Функция для заполнения строк со значениями NaN
def fill_nan_rows(chunk):
    chunk_filled = chunk.copy()
    nan_ranges = []
    for i, row in chunk[v_columns].iterrows():
        if row.isna().all():
            if i == 0:
                next_row = chunk.loc[i + 1, v_columns].astype(float)
                avg_values = next_row
            elif i == len(chunk) - 1:
                prev_row = chunk.loc[i - 1, v_columns].astype(float)
                avg_values = prev_row
            else:
                prev_row = chunk.loc[i - 1, v_columns].astype(float)
                next_row = chunk.loc[i + 1, v_columns].astype(float)
                avg_values = (prev_row + next_row) / 2
            chunk_filled.loc[i, v_columns] = avg_values
            nan_ranges.append((chunk.loc[i, 'batch_time'], avg_values))
    return chunk_filled, nan_ranges

# Создание объекта для записи очищенных данных
nan_periods = []
with open(output_file_path, 'w') as output_file:
    # Инициализация CSV ридера и встраивание заголовка
    with pd.read_csv(input_file_path, chunksize=chunksize) as reader:
        for i, chunk in enumerate(tqdm(reader, total=total_lines // chunksize, desc="Processing")):
            chunk = chunk.astype({col: 'float64' for col in v_columns})  # Приведение типов
            cleaned_chunk, nan_ranges = fill_nan_rows(chunk)
            nan_periods.extend(nan_ranges)
            if i == 0:
                cleaned_chunk.to_csv(output_file, index=False)  # Запись заголовков
            else:
                cleaned_chunk.to_csv(output_file, index=False, header=False)  # Без заголовков для следующих чанков

print(f"Очищенные данные сохранены в {output_file_path}")

# Вывод диапазонов, в которые были значения NaN
for start, values in nan_periods:
    print(f"Период простоя: {start} -> Средние значения: {values}")

# Визуализация (опционально)
import matplotlib.pyplot as plt

if nan_periods:
    times, values = zip(*nan_periods)
    plt.figure(figsize=(12, 6))
    plt.plot(times, [np.mean(v) for v in values], label="Средние значения")
    plt.title("Средние значения в периодах простоя")
    plt.xlabel("Время")
    plt.ylabel("Среднее значение")
    plt.legend()
    plt.grid(True)
    plt.show()
