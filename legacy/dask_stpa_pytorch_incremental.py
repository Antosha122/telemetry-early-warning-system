import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from dask.distributed import Client
import dask.dataframe as dd
from dask_ml.wrappers import Incremental
from sklearn.model_selection import train_test_split
import time
from dask import config

# Устанавливаем путь для временных файлов на другой диск
import tempfile

tempfile.tempdir = 'F:/tempp/'

# Устанавливаем клиент Dask для работы с данными
client = Client(processes=False, dashboard_address=':8792')


# Функция для определения наличия аварии через 3 часа
def check_emergency_in_3_hours(batch_time, emergency_dates):
    batch_datetime = datetime.strptime(batch_time, '%Y-%m-%d %H:%M:%S.%f')
    future_datetime = batch_datetime + timedelta(hours=3)
    return 1 if future_datetime in emergency_dates else 0


# Загрузка данных из файлов с использованием Dask
print("Loading data from files...")
opers_data = dd.read_csv('F:/ККАЛ/DataSet Газпром/opers.csv', blocksize='16MB')
stpa_data = dd.read_csv('F:/ККАЛ/DataSet Газпром/stpa.csv', blocksize='16MB')

# Фильтрация данных о наличии аварийных событий
print("Filtering emergency events data...")
emergency_dates = opers_data[opers_data['is_emergency']]['date'].compute().tolist()


# Обработка данных stpa для предсказания аварий через 3 часа
def process_stpa_data(chunk):
    # Фильтруем строки, где столбцы v_0 до v_3599 не заполнены нулями
    zero_mask = (chunk.iloc[:, 1:3600] != 0).any(axis=1)
    chunk_filtered = chunk[zero_mask]

    # Применяем функцию check_emergency_in_3_hours к столбцу batch_time
    chunk_filtered['emergency_in_3_hours'] = chunk_filtered['batch_time'].apply(
        lambda x: check_emergency_in_3_hours(x, emergency_dates), meta=('batch_time', 'int')
    )

    # Выбираем признаки и целевую переменную
    X = chunk_filtered.iloc[:, 1:-1]
    y = chunk_filtered['emergency_in_3_hours']
    return X, y


# Создаем трансформер для данных
transformer = FunctionTransformer(process_stpa_data)


# Создание модели PyTorch, обернутой в совместимый с sklearn класс
class PyTorchWrapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(PyTorchWrapper, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Уменьшаем размер слоя
        self.fc2 = nn.Linear(64, output_size)  # Используем один выходной слой
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def score(self, X, y):
        with torch.no_grad():
            predicted = (torch.sigmoid(self.forward(X)) > 0.5).float()
            return accuracy_score(y, predicted)


# Разделение данных на обучающий и тестовый наборы
print("Splitting data into training and testing sets...")
mask = (stpa_data.iloc[:, 1:3600] != 0).any(axis=1)
stpa_data_filtered = stpa_data[mask]

# Разделение на обучающую и тестовую выборки
train_stpa_data, test_stpa_data = train_test_split(stpa_data_filtered, test_size=0.2, random_state=42)

# Применяем трансформер к обучающим данным
X_train, y_train = transformer.transform(train_stpa_data)

# Создание объекта PyTorchWrapper для использования в Incremental
estimator = PyTorchWrapper(input_size=X_train.shape[1], output_size=1)

# Создание инкрементального обучающего объекта с использованием PyTorchWrapper
inc_train = Incremental(estimator, random_state=42)

# Обучение модели с оценкой
inc_train = inc_train.fit(X_train, y_train)

# Процесс поступления новых данных и анализа
while True:
    try:
        print("Введите данные для анализа (через запятую, без пробелов):")
        user_input = input().strip()

        # Преобразование пользовательского ввода в данные для анализа
        data_list = user_input.split(',')
        batch_time = data_list[0]
        features = list(map(float, data_list[1:]))  # Преобразование в числовой формат

        data_dict = {'batch_time': [batch_time], 'feature': features}
        data = pd.DataFrame(data_dict)

        # Применяем трансформер к новым данным
        X_new, _ = transformer.transform(data)

        # Применяем модель к новым данным
        predictions = inc_train.predict(X_new)

        # Выводим результаты анализа
        for prediction in predictions:
            if prediction == 1:
                print("Авария через 3 часа!")
            else:
                print("Авария не предсказывается через 3 часа.")

        # Ожидание следующей порции данных (например, с интервалом в 1 час)
        time.sleep(3600)  # ожидание 1 часа перед следующим анализом

    except Exception as e:
        print(f"Ошибка при анализе данных: {e}")









import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from dask.distributed import Client
import dask.dataframe as dd
from dask_ml.wrappers import Incremental
from sklearn.model_selection import train_test_split
import time
from dask import config

# Устанавливаем путь для временных файлов на другой диск
import tempfile

tempfile.tempdir = 'F:/tempp/'

# Устанавливаем клиент Dask для работы с данными
client = Client(processes=False, dashboard_address=':8792')


# Функция для определения наличия аварии через 3 часа
def check_emergency_in_3_hours(batch_time, emergency_dates):
    batch_datetime = datetime.strptime(batch_time, '%Y-%m-%d %H:%M:%S.%f')
    future_datetime = batch_datetime + timedelta(hours=3)
    return 1 if future_datetime in emergency_dates else 0


# Загрузка данных из файлов с использованием Dask
print("Loading data from files...")
opers_data = dd.read_csv('F:/ККАЛ/DataSet Газпром/opers.csv', blocksize='16MB')
stpa_data = dd.read_csv('F:/ККАЛ/DataSet Газпром/stpa.csv', blocksize='16MB')

# Фильтрация данных о наличии аварийных событий
print("Filtering emergency events data...")
emergency_dates = opers_data[opers_data['is_emergency']]['date'].compute().tolist()


# Обработка данных stpa для предсказания аварий через 3 часа
def process_stpa_data(chunk):
    # Фильтруем строки, где столбцы v_0 до v_3599 не заполнены нулями
    zero_mask = (chunk.iloc[:, 1:3600] != 0).any(axis=1)
    chunk_filtered = chunk[zero_mask]

    # Применяем функцию check_emergency_in_3_hours к столбцу batch_time
    chunk_filtered['emergency_in_3_hours'] = chunk_filtered['batch_time'].apply(
        lambda x: check_emergency_in_3_hours(x, emergency_dates), meta=('batch_time', 'int')
    )

    # Выбираем признаки и целевую переменную
    X = chunk_filtered.iloc[:, 1:-1]
    y = chunk_filtered['emergency_in_3_hours']
    return X, y

print("Создаем трансформер данных")
# Создаем трансформер для данных
transformer = FunctionTransformer(process_stpa_data)

print("# Создание модели PyTorch, обернутой в совместимый с sklearn класс")

class PyTorchWrapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(PyTorchWrapper, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Уменьшаем размер слоя
        self.fc2 = nn.Linear(64, output_size)  # Используем один выходной слой
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def score(self, X, y):
        with torch.no_grad():
            predicted = (torch.sigmoid(self.forward(X)) > 0.5).float()
            return accuracy_score(y, predicted)


# Разделение данных на обучающий и тестовый наборы
print("Splitting data into training and testing sets...")
mask = (stpa_data.iloc[:, 1:3600] != 0).any(axis=1)
stpa_data_filtered = stpa_data[mask]

print("Разделение на обучающую и тестовую выборки")
train_stpa_data, test_stpa_data = train_test_split(stpa_data_filtered, test_size=0.2, random_state=42)

print("Применяем трансформер к обучающим данным")
X_train, y_train = transformer.transform(train_stpa_data)

print("Создание объекта PyTorchWrapper для использования в Incremental")
estimator = PyTorchWrapper(input_size=X_train.shape[1], output_size=1)

print("Определяем функцию потерь и оптимизатор")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(estimator.parameters(), lr=0.001)

print("Обучение модели с оценкой")
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = estimator(X_train.float())
    loss = criterion(outputs, y_train.float().unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Model training complete.")

# Процесс поступления новых данных и анализа
while True:
    try:
        print("Введите данные для анализа (через запятую, без пробелов):")
        user_input = input().strip()

        # Преобразование пользовательского ввода в данные для анализа
        data_list = user_input.split(',')
        batch_time = data_list[0]
        features = list(map(float, data_list[1:]))  # Преобразование в числовой формат

        data_dict = {'batch_time': [batch_time], 'feature': features}
        data = pd.DataFrame(data_dict)

        # Применяем трансформер к новым данным
        X_new, _ = transformer.transform(data)

        # Применяем модель к новым данным
        predictions = inc_train.predict(X_new)

        # Выводим результаты анализа
        for prediction in predictions:
            if prediction == 1:
                print("Авария через 3 часа!")
            else:
                print("Авария не предсказывается через 3 часа.")

        # Ожидание следующей порции данных (например, с интервалом в 1 час)
        time.sleep(3600)  # ожидание 1 часа перед следующим анализом

    except Exception as e:
        print(f"Ошибка при анализе данных: {e}")
