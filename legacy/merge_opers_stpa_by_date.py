import pandas as pd
from tqdm import tqdm

def merge_files_by_date(stpa_file_path, opers_file_path, output_file_path, chunksize=500):
    print('Чтение файла opers')
    opers_data = pd.read_csv(opers_file_path, parse_dates=['date'])

    print(f"Количество строк в opers_data: {len(opers_data)}")
    print(f"Количество пропусков в opers_data:\n{opers_data.isnull().sum()}")
    print(f"Количество уникальных значений в opers_data:\n{opers_data.nunique()}")

    print('Чтение данных stpa')
    total_lines = sum(1 for _ in open(stpa_file_path)) - 1  # Минус 1 из-за заголовка
    print(f"Общее количество строк в stpa_file: {total_lines}")

    print("Объединение данных и отслеживание прогресса")

    is_first_chunk = True

    # Итерация по чанкам данных stpa
    with pd.read_csv(stpa_file_path, parse_dates=['batch_time'], chunksize=chunksize) as reader:
        for chunk in tqdm(reader, total=total_lines // chunksize, desc="Объединение данных"):
            chunk = chunk.rename(columns={'batch_time': 'date'})
            merged_chunk = pd.merge(opers_data, chunk, on='date', how='left')

            # Записываем в файл. Если это первый чанк, создаем файл, иначе дописываем.
            if is_first_chunk:
                merged_chunk.to_csv(output_file_path, index=False, mode='w', header=True)
                is_first_chunk = False
            else:
                merged_chunk.to_csv(output_file_path, index=False, mode='a', header=False)

    print(f"Объединение завершено. Результат сохранен в '{output_file_path}'.")

# Пути к файлам и размер чанка
stpa_file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'
opers_file_path = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
output_file_path = 'F:/ККАЛ/DataSet Газпром/merged_data_dask.csv'

# Вызов функции для объединения файлов с прогрессом
merge_files_by_date(stpa_file_path, opers_file_path, output_file_path)
