import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'
chunksize = 10000  # Размер блока для чтения файла


# Функция для анализа простоев
def analyze_downtime(chunk):
    downtime_info = []
    nan_periods = []
    start_time = None
    values_in_period = []

    for _, row in chunk.iterrows():
        if row.iloc[3:3603].isna().all():
            # Если это начало периода простоя
            if start_time is None:
                start_time = row['batch_time']
            # Собираем значения
            values_in_period.append(row.iloc[3:3603].values)
        else:
            # Если период простоя закончился
            if start_time is not None:
                avg_values = np.nanmean(values_in_period, axis=0)
                nan_periods.append((start_time, avg_values))
                start_time = None
                values_in_period = []

    # Если период простоя продолжается до конца чанка
    if start_time is not None:
        avg_values = np.nanmean(values_in_period, axis=0)
        nan_periods.append((start_time, avg_values))

    for period in nan_periods:
        downtime_info.append({
            'batch_time': period[0],
            'average_values': period[1]
        })
    return downtime_info


downtime_results = []

print("Анализ файла для определения простоев...")
with pd.read_csv(file_path, chunksize=chunksize, parse_dates=['batch_time']) as reader:
    for chunk in reader:
        downtime_info = analyze_downtime(chunk)
        downtime_results.extend(downtime_info)

# Преобразование результатов в DataFrame для дальнейшего анализа
downtime_df = pd.DataFrame(downtime_results)

# Сохранение результатов в файл для последующего анализа
output_file_path = 'F:/ККАЛ/DataSet Газпром/downtime_analysis.csv'
downtime_df.to_csv(output_file_path, index=False)

print("Анализ завершен. Результаты сохранены в:", output_file_path)

# Вывод диапазонов, в которые были значения NaN
print("Периоды простоя и средние значения в этих периодах:")
for _, row in downtime_df.iterrows():
    print(f"Период простоя: {row['batch_time']} -> Средние значения: {row['average_values']}")

# Визуализация (опционально)
if not downtime_df.empty:
    times = downtime_df['batch_time']
    avg_values = [np.mean(values) for values in downtime_df['average_values']]
    plt.figure(figsize=(12, 6))
    plt.plot(times, avg_values, label="Средние значения")
    plt.title("Средние значения в периодах простоя")
    plt.xlabel("Время")
    plt.ylabel("Среднее значение")
    plt.legend()
    plt.grid(True)
    plt.show()
