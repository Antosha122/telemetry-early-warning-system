# import pandas as pd
#
# file_path = 'F:/ККАЛ/DataSet Газпром/stpa.csv'
# chunksize = 10000  # Размер блока для чтения файла
#
# years = set()
# months = set()
# days = set()
#
# print("Анализ уникальных дат по столбцу 'batch_time'...")
# with pd.read_csv(file_path, chunksize=chunksize) as reader:
#     for chunk in reader:
#         # Преобразование столбца 'batch_time' в тип datetime
#         chunk['batch_time'] = pd.to_datetime(chunk['batch_time'])
#
#         # Получение уникальных значений годов, месяцев и дней
#         years.update(chunk['batch_time'].dt.year.unique())
#         months.update(chunk['batch_time'].dt.month.unique())
#         days.update(chunk['batch_time'].dt.day.unique())
#
# print("Уникальные года:", sorted(years))
# print("Уникальные месяцы:", sorted(months))
# print("Уникальные дни:", sorted(days))


import pandas as pd

file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'
chunksize = 10000  # Размер блока для чтения файла
print_interval = 100000  # Интервал строк для вывода

row_counter = 0  # Счетчик строк

print("Чтение файла и вывод строк каждые 100000 строк...")
with pd.read_csv(file_path, chunksize=chunksize) as reader:
    for chunk in reader:
        for _, row in chunk.iterrows():
            if row_counter % print_interval == 0:
                print(row)
            row_counter += 1

print("Завершено.")
