import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import time

start_time = time.time()


# Функция для проверки наличия чрезвычайного события через 3 часа
def check_emergency_in_3_hours(batch_time, emergency_dates):
    batch_datetime = datetime.strptime(batch_time, '%Y-%m-%d %H:%M:%S')
    future_datetime = batch_datetime + timedelta(hours=3)
    return 1 if future_datetime in emergency_dates else 0


try:
    print('Загрузка файла opers...')
    # Чтение данных из файла opers.csv чанками
    file_path_opers = 'F:/ККАЛ/DataSet Газпром/opers.csv'
    n_rows_to_load_opers = 500000  # Загрузить только первые 100000 строк
    chunksize_opers = 10000  # Размер чанка для чтения данных
    opers_data_chunks = pd.read_csv(file_path_opers, chunksize=chunksize_opers)
    # Переменные для хранения обработанных данных
    emergency_dates = set()
    opers_data = pd.DataFrame()
    # Обработка данных по частям
    for i, chunk in enumerate(opers_data_chunks):
        # Обработка только первых n_rows_to_load_opers строк
        if len(opers_data) >= n_rows_to_load_opers:
            break
        # Добавление части данных к общему DataFrame
        opers_data = pd.concat([opers_data, chunk], ignore_index=True)
        print(f"Загружено {len(opers_data)} строк данных из файла opers.csv.")
    # Фильтрация данных о чрезвычайных событиях
    filtered_opers_data = opers_data[opers_data['is_emergency'] == True]
    # Сохранение отфильтрованных данных в новый файл
    filtered_file_path = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
    filtered_opers_data.to_csv(filtered_file_path, index=False)
    print(f"Сохранены отфильтрованные данные в файл: {filtered_file_path}")
    # Получение множества дат чрезвычайных событий
    emergency_dates.update(filtered_opers_data['date'])
    print(f"Загружено {len(filtered_opers_data)} чрезвычайных событий из файла opers.csv.")

    # Загрузка файла stpa
    try:
        print('Загрузка файла stpa...')
        file_path_stpa = 'F:/ККАЛ/DataSet Газпром/stpa.csv'
        n_rows_to_load_stpa = 500000  # Загрузить только первые 500000 строк
        chunksize_stpa = 100000  # Размер чанка для чтения данных
        stpa_data_chunks = pd.read_csv(file_path_stpa, chunksize=chunksize_stpa, engine='python')
        stpa_filtered_data = pd.DataFrame()  # Переменная для хранения обработанных данных
        # Обработка данных по частям
        for i, chunk in enumerate(stpa_data_chunks):
            # Обработка только первых n_rows_to_load_stpa строк
            if len(stpa_filtered_data) >= n_rows_to_load_stpa:
                break
            # Фильтрация данных о чрезвычайных событиях в файле stpa.csv
            filtered_chunk = chunk[chunk['batch_time'].isin(filtered_opers_data)]
            stpa_filtered_data = pd.concat([stpa_filtered_data, filtered_chunk], ignore_index=True)
            # Сохранение отфильтрованных данных в новый файл
            filtered_file_path_stpa = 'F:/ККАЛ/DataSet Газпром/filtered_stpa.csv'
            stpa_filtered_data.to_csv(filtered_file_path_stpa, index=False)
            print(f"Сохранены отфильтрованные данные из файла stpa в файл: {filtered_file_path_stpa}")
            print(f"Загружено {len(stpa_filtered_data)} чрезвычайных событий из файла stpa.csv.")
    except Exception as e:
        print(f"Ошибка загрузки или обработки файла 'stpa.csv': {e}")

    print(f"Размер исходного набора данных: {len(stpa_filtered_data)}")

    # Разделение данных на обучающий и тестовый наборы
    if len(stpa_filtered_data) > 1000:
        train_stpa_data, test_stpa_data = train_test_split(stpa_filtered_data, test_size=0.2, random_state=48)


        # Функция для обработки данных stpa_data
        def process_stpa_data(chunk):
            zero_mask = (chunk.iloc[:, 1:3600] != 0).all(axis=1)
            chunk_filtered = chunk[zero_mask]
            chunk_filtered['emergency_in_3_hours'] = chunk_filtered['batch_time'].apply(
                lambda x: check_emergency_in_3_hours(x, emergency_dates)
            )
            X = chunk_filtered.iloc[:, 1:-1]
            y = chunk_filtered['emergency_in_3_hours']
            return X, y


        transformer = FunctionTransformer(process_stpa_data)
        X_train, y_train = transformer.transform(train_stpa_data)
        X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').dropna()
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_numeric)
        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)


        class PyTorchWrapper(nn.Module):
            def __init__(self, input_size, output_size):
                super(PyTorchWrapper, self).__init__()
                self.fc1 = nn.Linear(input_size, 512)
                self.fc2 = nn.Linear(512, 512)
                self.fc3 = nn.Linear(512, 512)
                self.fc4 = nn.Linear(512, 512)
                self.fc5 = nn.Linear(512, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                x = self.relu(self.fc4(x))
                x = self.fc5(x)
                return x


        estimator = PyTorchWrapper(input_size=X_train_tensor.shape[1], output_size=1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.01, weight_decay=0.001)

        print("Обучение модели...")
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = estimator(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.unsqueeze(1))
            loss.backward()
            optimizer.step()
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {loss.item()}")
        print("Обучение модели завершено.")

        torch.save(estimator.state_dict(), 'trained_model.pth')
        print("Обученная модель сохранена.")

        loaded_model = PyTorchWrapper(input_size=X_train_tensor.shape[1], output_size=1)
        loaded_model.load_state_dict(torch.load('trained_model.pth'))
        loaded_model.eval()

        while True:
            try:
                print("Введите данные для анализа (разделенные запятой без пробелов):")
                user_input = input().strip()
                data_list = user_input.split(',')
                batch_time = data_list[0]
                features = list(map(float, data_list[1:]))
                data_dict = {'batch_time': [batch_time], 'feature': features}
                data = pd.DataFrame(data_dict)
                X_new, _ = transformer.transform(data)
                X_new_numeric = X_new.apply(pd.to_numeric, errors='coerce').dropna()
                X_new_normalized = scaler.transform(X_new_numeric)
                predictions = loaded_model(torch.tensor(X_new_normalized, dtype=torch.float32))
                for prediction in predictions:
                    if prediction.item() > 0.5:
                        print("Чрезвычайное событие прогнозируется в течение 3 часов!")
                    else:
                        print("Чрезвычайное событие не прогнозируется в течение 3 часов.")
                time.sleep(3600)
            except Exception as e:
                print(f"Ошибка во время анализа данных: {e}")
    else:
        print("Исходный набор данных слишком маленький для разделения.")
except Exception as e:
    print(f"Ошибка загрузки или обработки данных: {e}")

end_time = time.time()
execution_time = end_time - start_time
print(f"Общее время выполнения: {execution_time:.2f} секунд")
