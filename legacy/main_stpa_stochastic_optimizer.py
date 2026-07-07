import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import time
import os
# Определите переменную device для использования CUDA (GPU) или CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Определение нового оптимизатора StochasticAdaptiveOptimizer
class StochasticAdaptiveOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super(StochasticAdaptiveOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['avg_squared_grad'] = torch.zeros_like(p.data)

                avg_squared_grad = state['avg_squared_grad']
                beta = group['beta']
                epsilon = group['epsilon']

                state['step'] += 1

                avg_squared_grad.mul_(beta).addcmul_(1 - beta, grad, grad)
                denom = avg_squared_grad.sqrt().add_(epsilon)
                adaptive_lr = group['lr'] / denom

                p.data.add_(-adaptive_lr, grad)

        return loss

# Определение функций для работы с данными
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
    if all(value == 0 for value in new_data):
        print("Нет информации о возможной аварии.")
        return

    with torch.no_grad():
        features = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(features)
        prediction = torch.sigmoid(outputs) > 0.5

        if prediction.item() == 1:
            print("Вероятность аварии через 3 часа высока.")
        else:
            print("Вероятность аварии через 3 часа низка.")

def check_file(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0

# Загрузка и обработка данных
print("Загрузка и обработка данных...")

opers_filename = r'F:\ККАЛ\DataSet Газпром\opers.csv'
stpa_filename = r'F:\ККАЛ\DataSet Газпром\stpa.csv'

if not (check_file(opers_filename) and check_file(stpa_filename)):
    print("Ошибка: Один из файлов не существует или пуст.")
    exit()

opers_data = pd.read_csv(opers_filename)
stpa_data = pd.read_csv(stpa_filename)

if opers_data.shape[0] != stpa_data.shape[0]:
    print("Ошибка: Несовпадение размеров данных в файлах opers.csv и stpa.csv.")
    exit()

filename = 'merged.dat'
merged_memmap = np.memmap(filename, dtype='float32', mode='w+', shape=(0, 3601))

for i in range(opers_data.shape[0]):
    if opers_data.iloc[i, 0] == stpa_data.iloc[i, 0]:
        merged_data = np.concatenate((opers_data.iloc[i, :].values, stpa_data.iloc[i, -1:]))
        merged_memmap.resize((merged_memmap.shape[0] + 1, 3601))
        merged_memmap[-1] = merged_data

print("Удаление ненужных столбцов и заполнение пропущенных значений...")
merged_memmap.fill(0)

# Преобразование данных в тензоры для PyTorch
print("Преобразование данных в тензоры для PyTorch...")
X = torch.tensor(merged_memmap[:, :-1], dtype=torch.float32).to(device)
y = torch.tensor(merged_memmap[:, -1], dtype=torch.float32).unsqueeze(1).to(device)

# Создание DataLoader
print("Создание DataLoader...")
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Определение модели нейронной сети
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
input_size = 3600
hidden_size = 256
output_size = 1
model = Model(input_size, hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = StochasticAdaptiveOptimizer(model.parameters(), lr=0.001)  # Используем новый оптимизатор
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Обучение модели
print("Обучение модели...")
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()}, Loss: {loss.item()}')

print("Обучение завершено.")

# Ожидание новых данных и анализ
while True:
    new_data = get_new_data()
    analyze_new_data(new_data, model)
    time.sleep(3599)  # Пауза на 1 час (3599 секунд)
