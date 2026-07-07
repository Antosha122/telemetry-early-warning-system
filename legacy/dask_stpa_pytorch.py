import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import dask.dataframe as dd
import pandas as pd

# Устройство CUDA (GPU), если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_new_data():
    while True:
        try:
            new_data_str = input("Введите новые данные (значения): ")
            values = process_input_data(new_data_str)

            if len(values) != 3600:
                print("Неверное количество значений. Пожалуйста, введите ровно 3600 значений.")
                continue

            return values
        except ValueError as e:
            print("Ошибка:", e)
            print("Неверный формат данных. Пожалуйста, введите корректные данные.")


def process_input_data(new_data):
    data_parts = new_data.split(',')

    while len(data_parts) < 3600:
        data_parts.append('0')

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
print('Загрузка данных')
# Чтение данных из файла stpla.csv с использованием Dask
stpla_df = dd.read_csv('F:/ККАЛ/DataSet Газпром/stpa.csv', dtype='str')

# Преобразование данных в числовой формат с обработкой ошибок
def convert_to_float(series):
    return pd.to_numeric(series, errors='coerce').astype('float32')

# Применение преобразования к столбцам начинающимся с 'v_'
for column in stpla_df.columns[stpla_df.columns.str.startswith('v_')]:
    stpla_df[column] = stpla_df[column].map_partitions(convert_to_float)

# Вычисление выражений Dask
stpla_df = stpla_df.compute()

# Чтение данных из файла opers.csv с использованием pandas
opers_df = pd.read_csv('F:/ККАЛ/DataSet Газпром/opers.csv')

# Объединение данных по индексам
merged_df = pd.merge(opers_df, stpla_df, on='index_column', how='inner')

print("Преобразование данных в тензоры для PyTorch...")
# Преобразование данных в тензоры для PyTorch
X = torch.tensor(merged_df[['feature_columns']].values.compute(), dtype=torch.float32).to(device)
y = torch.tensor(merged_df['target_column'].values.compute(), dtype=torch.float32).to(device)

print("Создание DataLoader...")
# Создание DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Определение архитектуры нейронной сети
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Гиперпараметры модели
input_size = 3600
hidden_size = 128
output_size = 1

print("Инициализация модели...")
# Инициализация модели
model = Model(input_size, hidden_size, output_size).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Планировщик learning rate
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

print("Начало обучения...")
# Обучение модели
total_processed_data = 0
num_epochs = 100

for epoch in range(num_epochs):
    processed_data_this_epoch = 0

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

        # Увеличение счетчиков обработанных данных
        processed_data_this_epoch += len(inputs)
        total_processed_data += len(inputs)

    # Уменьшение learning rate
    scheduler.step()

    # Вывод текущего значения learning rate и количества обработанных данных
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()}, Loss: {loss.item()}, Processed Data This Epoch: {processed_data_this_epoch}')

print("Обучение завершено.")
print(f"Всего обработано данных: {total_processed_data}")

print("Ожидание новых данных для анализа...")
# Бесконечный цикл для мониторинга новых данных
while True:
    new_data = get_new_data()
    analyze_new_data(new_data, model)
