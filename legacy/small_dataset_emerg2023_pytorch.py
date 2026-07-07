import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# Устройство CUDA (GPU), если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_new_data():
    while True:
        try:
            # Ввод данных в формате: "value1, value2, ..."
            new_data_str = input("Введите новые данные (значения): ")
            values = process_input_data(new_data_str)  # Обработка введенных данных

            # Проверка наличия всех элементов в значении
            if len(values) != 3600:
                print("Неверное количество значений. Пожалуйста, введите ровно 3600 значений.")
                continue

            return values
        except ValueError as e:
            print("Ошибка:", e)
            print("Неверный формат данных. Пожалуйста, введите корректные данные.")

def process_input_data(new_data):
    # Разделение введенной строки по запятой
    data_parts = new_data.split(',')

    # Заполнение пропущенных значений нулями
    while len(data_parts) < 3600:
        data_parts.append('0')

    # Преобразование значений в тензор и нормализация
    values = [(float(value) - 0.1) / 0.9 for value in data_parts]

    return values

def analyze_new_data(new_data, model):
    # Проверка, что все значения в новых данных равны нулю
    if all(value == 0 for value in new_data):
        print("Нет информации о возможной аварии.")
        return

    with torch.no_grad():
        # Подготовка признаков для передачи в модель
        features = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Предсказание с помощью обученной модели
        outputs = model(features)
        prediction = torch.sigmoid(outputs) > 0.5

        # Вывод результата предсказания
        if prediction.item() == 1:
            print("Вероятность аварии через 3 часа высока.")
        else:
            print("Вероятность аварии через 3 часа низка.")

# Загрузка данных из файлов и преобразование их
print("Загрузка данных...")
opers_data = pd.read_csv('emerg2023.csv')
stpla_data = pd.read_csv('stpa5000.csv')

# Приведение типа данных столбцов v_0, v_1, ..., v_3599 к числовым значениям (float)
for column in stpla_data.columns[stpla_data.columns.str.startswith('v_')]:
    stpla_data[column] = pd.to_numeric(stpla_data[column], errors='coerce')

# Объединение данных по индексам
print("Объединение данных...")
merged_data = pd.merge(opers_data, stpla_data, left_index=True, right_index=True, how='inner')

# Удаление ненужных столбцов и заполнение пропущенных значений
print("Удаление ненужных столбцов и заполнение пропущенных значений...")
merged_data.fillna(0, inplace=True)

# Преобразование временных данных в формат datetime
print("Преобразование временных данных...")
merged_data['date'] = pd.to_datetime(merged_data['date'])

# Преобразование данных в тензоры для PyTorch
print("Преобразование данных в тензоры для PyTorch...")
v_features = [f'v_{i}' for i in range(3600)]
X = torch.tensor(merged_data[v_features].values, dtype=torch.float32).to(device)
y = torch.tensor(merged_data['is_emergency'].values, dtype=torch.float32).to(device)

# Создание DataLoader
print("Создание DataLoader...")
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Определение архитектуры нейронной сети
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Гиперпараметры модели
input_size = 3600
hidden_size = 64
output_size = 1

# Инициализация модели
print("Инициализация модели...")
model = Model(input_size, hidden_size, output_size).to(device)

# Определение функции потерь и оптимизатора
print("Определение функции потерь и оптимизатора...")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
print("Обучение модели...")
num_epochs = 50
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

print("Обучение завершено.")

# Бесконечный цикл для мониторинга новых данных
while True:
    # Получение новых данных
    new_data = get_new_data()

    # Анализ новых данных и вывод времени возможной аварии за 3 часа до ее возникновения
    analyze_new_data(new_data, model)

    # Ждем 1 час перед получением новых данных
    time.sleep(3599)  # Время в секундах
