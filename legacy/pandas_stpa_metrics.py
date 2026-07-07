import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler


def normalize_data(data):
    # Удаление строк с пустыми значениями (NaN)
    clean_data = data.dropna()
    # Проверка, остались ли строки после удаления
    if not clean_data.empty:
        # Нормализация данных (пример: масштабирование значений)
        # Например, можно использовать Min-Max нормализацию:
        clean_data.iloc[:, 3:] = (clean_data.iloc[:, 3:] - clean_data.iloc[:, 3:].min()) / (
                clean_data.iloc[:, 3:].max() - clean_data.iloc[:, 3:].min())
    return clean_data

# Загрузка данных из файла opers.csv с отображением прогресса
file_path_opers = 'F:/ККАЛ/DataSet Газпром/opers.csv'
print("Loading data from:", file_path_opers)
opers_data = pd.read_csv(file_path_opers)

# Преобразование столбца 'date' в формат datetime, если он еще не преобразован
with tqdm(total=len(opers_data), desc='Converting date to datetime', unit='rows') as pbar:
    opers_data['date'] = pd.to_datetime(opers_data['date'])
    pbar.update(len(opers_data))

# Фильтрация данных по столбцу 'is_emergency' с значениями 'True' и 'False'
filtered_data = opers_data[opers_data['is_emergency'].isin([True, False])]

# Получение дат, когда была авария (когда is_emergency = True)
emergency_dates = filtered_data['date']
filtered_data = filtered_data[['is_emergency', 'date']]

# Сохранение отфильтрованных данных в новый файл с отображением прогресса
output_file_path_filtered = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
print("Saving filtered data to:", output_file_path_filtered)
with tqdm(total=len(filtered_data), desc='Saving filtered data', unit='rows') as pbar:
    filtered_data.to_csv(output_file_path_filtered, index=False)
    pbar.update(len(filtered_data))

# Вывод информации о датах аварии
print("Dates of emergencies:")
print(emergency_dates)

# Загрузка данных из файла stpa.csv с отображением прогресса
file_path_stpa = 'F:/ККАЛ/DataSet Газпром/stpa.csv'
chunksize = 10000

# Открытие файла для чтения и обработки данных с отображением прогресса
print("Loading data from:", file_path_stpa)
# Открытие файла для чтения и обработки данных с отображением прогресса
with tqdm(desc='Processing data chunks', unit='chunk') as pbar:
    processed_data = []  # Определение списка для хранения обработанных данных
    for chunk in pd.read_csv(file_path_stpa, chunksize=chunksize):
        # Удаление строк с пустыми значениями (NaN)
        clean_chunk = chunk.dropna(how='all')
        # Проверка, остались ли строки после удаления
        if not clean_chunk.empty:
            processed_chunk = normalize_data(clean_chunk)
            processed_data.append(processed_chunk)
            pbar.update(1)  # Обновляем прогресс

# Соединяем все чанки в один DataFrame с отображением прогресса
processed_df = pd.concat(processed_data)
processed_file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'

print("Saving processed data to:", processed_file_path)
with tqdm(total=len(processed_df), desc='Saving processed data', unit='rows') as pbar:
    processed_df.to_csv(processed_file_path, index=False)
    pbar.update(len(processed_df))

print("Normalization and cleaning completed. Result saved to:", processed_file_path)

# Загрузка данных из файла filtered_opers.csv
filtered_opers_file_path = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
print("Loading data from:", filtered_opers_file_path)
filtered_opers_data = pd.read_csv(filtered_opers_file_path)

# Преобразование столбца 'date' в тип данных datetime64[ns]
filtered_opers_data['date'] = pd.to_datetime(filtered_opers_data['date'])

# Загрузка данных из файла normalized_stpa.csv
normalized_stpa_file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'

# Определяем общее количество строк в файле normalized_stpa.csv для отображения прогресса
total_rows = sum(1 for line in open(normalized_stpa_file_path))

print("Loading data from:", normalized_stpa_file_path)
# Используем tqdm для отображения прогресса при чтении файла normalized_stpa.csv
with tqdm(total=total_rows, desc='Loading normalized_stpa.csv', unit='rows') as pbar:
    normalized_stpa_data = pd.read_csv(normalized_stpa_file_path, chunksize=chunksize)
    processed_data = []
    for chunk in normalized_stpa_data:
        # Обработка данных как ранее
        # Удаление строк с пустыми значениями (NaN)
        clean_chunk = chunk.dropna()
        # Нормализация данных (пример: масштабирование значений)
        # Например, можно использовать Min-Max нормализацию:
        clean_chunk.iloc[:, 3:] = (clean_chunk.iloc[:, 3:] - clean_chunk.iloc[:, 3:].min()) / (
                    clean_chunk.iloc[:, 3:].max() - clean_chunk.iloc[:, 3:].min())
        # Добавляем очищенный чанк к результату
        processed_data.append(clean_chunk)
        # Обновляем индикатор прогресса
        pbar.update(len(chunk))

# Преобразование столбца 'batch_time' в формат datetime для каждого DataFrame
for idx, chunk in enumerate(processed_data):
    processed_data[idx]['batch_time'] = pd.to_datetime(chunk['batch_time'])

# Соединяем все чанки в один DataFrame
normalized_stpa_data = pd.concat(processed_data)

# Выборка уникальных дат из отфильтрованных данных
emergency_dates = filtered_opers_data['date']

# Фильтрация данных из normalized_stpa.csv по датам аварий
merged_data = pd.merge(normalized_stpa_data, filtered_opers_data, left_on='batch_time', right_on='date', how='inner')

# Путь для сохранения объединенных данных
merged_file_path = 'F:/ККАЛ/DataSet Газпром/merged_data.csv'

# Сохранение объединенных данных в новый файл
merged_data.to_csv(merged_file_path, index=False)

print("Merging completed. Result saved to:", merged_file_path)


print('Загрузка данных для обучения из файла merged_file_path')
data = pd.read_csv(merged_file_path)

# Разделение данных на признаки (X) и метки (y)
X = data.iloc[:, 3:3602].values
y = data['is_emergency'].values

print('Разделение данных на обучающий и тестовый наборы')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Определение архитектуры нейронной сети
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = NeuralNet(input_size=X.shape[1])

# Step 1: Check for Class Imbalance
print(f"Distribution of 'is_emergency':")
print(data['is_emergency'].value_counts())

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

class_counts = np.bincount(y_train_resampled)
class_weights = torch.tensor([class_counts[0] / len(y_train_resampled), class_counts[1] / len(y_train_resampled)], dtype=torch.float32)
criterion = nn.BCELoss(weight=class_weights)

# Определение функции потерь и оптимизатора
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 100
batch_size = 64
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Прямой проход
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Обратное распространение и оптимизация
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed.")

# Оценка модели
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).round()
    y_pred_test = model(X_test_tensor).round()

train_accuracy = accuracy_score(y_train, y_pred_train.numpy())
test_accuracy = accuracy_score(y_test, y_pred_test.numpy())

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Вычисление метрик
precision = precision_score(y_test, y_pred_test.numpy())  # Точность показывает, какую долю объектов, предсказанных моделью как положительные, действительно являются положительными.
recall = recall_score(y_test, y_pred_test.numpy())  # Полнота показывает, какую долю положительных объектов модель правильно предсказала.
f1 = f1_score(y_test, y_pred_test.numpy())  # F1-Score является гармоническим средним точности и полноты. Это полезная метрика, если вам нужно найти баланс между точностью и полнотой.
roc_auc = roc_auc_score(y_test, y_pred_test.numpy())    #ROC-AUC измеряет способность модели различать положительные и отрицательные классы.
logloss = log_loss(y_test, y_pred_test.numpy()) # Log-Loss измеряет производительность классификационной модели, где выход является вероятностью от 0 до 1. Это вычисляется как отрицательное логарифмическое правдоподобие.
mcc = matthews_corrcoef(y_test, y_pred_test.numpy())    #MCC учитывает все элементы матрицы ошибок и является сбалансированной метрикой, подходящей даже для несбалансированных наборов данных.
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test.numpy()).ravel()  #Confusion Matrix (Матрица ошибок)
specificity = tn / (tn + fp)    #Специфичность измеряет способность модели правильно идентифицировать отрицательные примеры.

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')
print(f'Log-Loss: {logloss:.2f}')
print(f'Matthews Correlation Coefficient: {mcc:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'True Positives (TP): {tp}')
print(f'False Positives (FP): {fp}')
print(f'True Negatives (TN): {tn}')
print(f'False Negatives (FN): {fn}')

# Сохранение обученной модели
torch.save(model.state_dict(), 'path_to_save_model.pth')
print("Model saved.")

    # Проверка достижения целевой точности
    # if accuracy >= target_accuracy:
    #     print(f'Target accuracy of {target_accuracy} achieved. Stopping training.')
    #     break


