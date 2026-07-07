import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os
import gc
import concurrent.futures
#
# def normalize_data(data):
#     clean_data = data.dropna()
#     if not clean_data.empty:
#         clean_data.iloc[:, 3:] = (clean_data.iloc[:, 3:] - clean_data.iloc[:, 3:].min()) / (
#                 clean_data.iloc[:, 3:].max() - clean_data.iloc[:, 3:].min())
#     return clean_data

class LargeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def normalize_chunk(chunk, scaler):
    return scaler.transform(chunk)

def normalize_data_parallel(X_train_resampled, n_jobs=None):
    chunk_size = 10000
    scaler = StandardScaler()
    scaler.fit(X_train_resampled)
    chunks = [X_train_resampled[i:i + chunk_size] for i in range(0, len(X_train_resampled), chunk_size)]
    if n_jobs is None:
        n_jobs = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda chunk: normalize_chunk(chunk, scaler), chunks))
    X_train_scaled = np.vstack(results)
    return X_train_scaled


#
# # Загрузка данных из файла opers.csv с отображением прогресса
# file_path_opers = 'F:/ККАЛ/DataSet Газпром/opers.csv'
# print("Loading data from:", file_path_opers)
# opers_data = pd.read_csv(file_path_opers)
#
# # Удаление ненужных столбцов
# cols_to_drop = ['svod_opers_id', 'op_templ_id', 'r_id', 'duration_hours']
# opers_data.drop(columns=cols_to_drop, inplace=True)
#
# # Преобразование столбца 'date' в формат datetime
# print("Converting 'date' column to datetime...")
# with tqdm(total=len(opers_data), desc='Converting date to datetime', unit='rows') as pbar:
#     opers_data['date'] = pd.to_datetime(opers_data['date'])
#     pbar.update(len(opers_data))
#
# # Фильтрация данных по столбцу 'is_emergency'
# print("Filtering data by 'is_emergency' column...")
# filtered_data = opers_data[['is_emergency', 'date']]
#
# # Сохранение отфильтрованных данных
# output_file_path_filtered = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
# print("Saving filtered data to:", output_file_path_filtered)
# with tqdm(total=len(filtered_data), desc='Saving filtered data', unit='rows') as pbar:
#     filtered_data.to_csv(output_file_path_filtered, index=False)
#     pbar.update(len(filtered_data))
#
# # Загрузка данных из файла stpa.csv
# file_path_stpa = 'F:/ККАЛ/DataSet Газпром/stpa.csv'
# chunksize = 10000
#
# print("Loading data from:", file_path_stpa)
# processed_file_path_stpa = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'
# processed_data = []
# header_saved = False
#
# with open(processed_file_path_stpa, 'w') as f:
#     pass
#
# with tqdm(desc='Processing stpa data chunks', unit='chunk') as pbar:
#     for i, chunk in enumerate(pd.read_csv(file_path_stpa, chunksize=chunksize, low_memory=True, parse_dates=['batch_time'])):
#         # Удаляем первые два столбца
#         chunk = chunk.iloc[:, 2:]
#
#         # Удаление строк, где все значения NaN
#         clean_chunk = chunk.dropna(how='all')
#
#         # Заполнение пропусков средними значениями столбцов
#         clean_chunk = clean_chunk.fillna(clean_chunk.mean())
#
#         if not clean_chunk.empty:
#             processed_chunk = normalize_data(clean_chunk)
#             processed_data.append(processed_chunk)
#
#             # Сохраняем данные каждые 5 чанков
#             if (i + 1) % 5 == 0:
#                 pd.concat(processed_data).to_csv(processed_file_path_stpa, mode='a', header=not header_saved,index=False)
#                 header_saved = True
#                 processed_data = []
#                 gc.collect()
#
#         pbar.update(1)
#
# # Сохраняем оставшиеся данные
# if processed_data:
#     pd.concat(processed_data).to_csv(processed_file_path_stpa, mode='a', header=not header_saved, index=False)
#
# print("Normalized stpa data saved to:", processed_file_path_stpa)
#
# # Освобождение памяти
# del processed_data
# gc.collect()

# Загрузка данных
filtered_opers_file_path = 'F:/ККАЛ/DataSet Газпром/filtered_opers.csv'
print("Загрузка данных из:", filtered_opers_file_path)
filtered_opers_data = pd.read_csv(filtered_opers_file_path)

# Преобразование столбца 'date' в формат datetime и установка точности до минуты
filtered_opers_data['date'] = pd.to_datetime(filtered_opers_data['date']).dt.floor('min')

# Загрузка данных из файла normalized_stpa.csv
normalized_stpa_file_path = 'F:/ККАЛ/DataSet Газпром/normalized_stpa.csv'
chunksize = 10000
total_rows = sum(1 for line in open(normalized_stpa_file_path))

print("Загрузка данных из:", normalized_stpa_file_path)
with tqdm(total=total_rows, desc='Загрузка normalized_stpa.csv', unit='rows') as pbar:
    normalized_stpa_data = pd.read_csv(normalized_stpa_file_path, chunksize=chunksize)
    processed_data = []
    for chunk in normalized_stpa_data:
        clean_chunk = chunk.dropna()
        clean_chunk.iloc[:, 3:] = (clean_chunk.iloc[:, 3:] - clean_chunk.iloc[:, 3:].mean()) / clean_chunk.iloc[:, 3:].std()

        # Преобразование столбца 'batch_time' в формат datetime и установка точности до минуты
        clean_chunk['batch_time'] = pd.to_datetime(clean_chunk['batch_time']).dt.floor('min')

        processed_data.append(clean_chunk)
        pbar.update(len(chunk))

print("Объединение всех чанков в один DataFrame...")
normalized_stpa_data = pd.concat(processed_data)

del processed_data
gc.collect()

# Проверка типов данных
print("Тип данных в 'batch_time':", normalized_stpa_data['batch_time'].dtype)
print("Тип данных в 'date':", filtered_opers_data['date'].dtype)

# Проверка количества уникальных значений
print("Количество уникальных дат в 'batch_time':", normalized_stpa_data['batch_time'].nunique())
print("Количество уникальных дат в 'date':", filtered_opers_data['date'].nunique())

# Преобразование столбца 'batch_time' в тип данных datetime64[ns]
normalized_stpa_data['batch_time'] = pd.to_datetime(normalized_stpa_data['batch_time'])

# Объединение данных на основе даты и времени с точностью до минуты
merged_data = pd.merge(normalized_stpa_data, filtered_opers_data, left_on='batch_time', right_on='date', how='inner')
print(f"Количество строк после объединения: {len(merged_data)}")

# Проверка распределения классов после объединения
print("Распределение классов после объединения:")
print(merged_data['is_emergency'].value_counts())

# Сохранение объединенных данных
merged_file_path = 'F:/ККАЛ/DataSet Газпром/merged_data.csv'
chunk_size = 10000

print("Сохранение объединенных данных в:", merged_file_path)
with tqdm(total=len(merged_data), desc='Сохранение объединенных данных', unit='rows') as pbar:
    with open(merged_file_path, 'w', newline='') as f:
        merged_data.iloc[:0].to_csv(f, index=False)
        for i in range(0, len(merged_data), chunk_size):
            merged_data.iloc[i:i + chunk_size].to_csv(f, header=False, index=False)
            pbar.update(min(chunk_size, len(merged_data) - i))

print("Объединение завершено. Результат сохранен в:", merged_file_path)

print('Загрузка объединенных данных для обучения...')
total_rows = sum(1 for line in open(merged_file_path)) - 1
data_chunks = []

with tqdm(total=total_rows, desc='Загрузка объединенных данных для обучения', unit='rows') as pbar:
    for chunk in pd.read_csv(merged_file_path, chunksize=chunk_size):
        data_chunks.append(chunk)
        pbar.update(len(chunk))

data = pd.concat(data_chunks)
print(f"Количество строк в загруженных данных: {len(data)}")

del data_chunks
gc.collect()

data = data.sample(frac=0.6, random_state=42)
print(f"Количество строк в уменьшенных данных: {len(data)}")

X = data.drop(columns=['batch_time', 'date', 'is_emergency']).to_numpy()
y = data['is_emergency'].to_numpy()
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Количество строк в уменьшенных данных: {len(X_reduced)}")

kf = KFold(n_splits=2, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in kf.split(X_reduced):
    print(f"Fold {fold}")
    X_train, X_test = X_reduced[train_index], X_reduced[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Балансировка данных: SMOTE и RandomUnderSampler
    smote = SMOTE(random_state=42)
    rus = RandomUnderSampler(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_resampled, y_train_resampled)

    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f'Распределение после балансировки: {dict(zip(unique, counts))}')

    X_train_scaled = normalize_data_parallel(X_train_resampled)
    scaler = StandardScaler().fit(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = LargeDataset(X_train_scaled, y_train_resampled)
    test_dataset = LargeDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class NeuralNet(nn.Module):
        def __init__(self, input_size, dropout_rate=0.5):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(p=dropout_rate)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(p=dropout_rate)
            self.fc3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.relu3 = nn.ReLU()
            self.drop3 = nn.Dropout(p=dropout_rate)
            self.fc4 = nn.Linear(128, 64)
            self.bn4 = nn.BatchNorm1d(64)
            self.relu4 = nn.ReLU()
            self.drop4 = nn.Dropout(p=dropout_rate)
            self.fc5 = nn.Linear(64, 32)
            self.bn5 = nn.BatchNorm1d(32)
            self.relu5 = nn.ReLU()
            self.drop5 = nn.Dropout(p=dropout_rate)
            self.fc6 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.drop2(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.drop3(x)
            x = self.fc4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.drop4(x)
            x = self.fc5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.drop5(x)
            x = self.fc6(x)
            x = self.sigmoid(x)
            return x

    input_size = X_train_scaled.shape[1]
    model = NeuralNet(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = 10
    best_f1_score = 0
    writer = SummaryWriter(log_dir=f'runs/fold_{fold}')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}')

        model.eval()
        all_labels = []
        all_predictions = []
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.float()
                labels = labels.float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                all_labels.extend(labels.numpy())
                all_predictions.extend(predictions.numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        fold_accuracy = accuracy_score(all_labels, all_predictions)
        fold_precision = precision_score(all_labels, all_predictions)
        fold_recall = recall_score(all_labels, all_predictions)
        fold_f1 = f1_score(all_labels, all_predictions)
        fold_mcc = matthews_corrcoef(all_labels, all_predictions)
        fold_roc_auc = roc_auc_score(all_labels, all_predictions)
        fold_log_loss = log_loss(all_labels, all_predictions)

        writer.add_scalar('Metrics/Test Loss', test_loss / len(test_loader), epoch)
        writer.add_scalar('Metrics/Accuracy', fold_accuracy, epoch)
        writer.add_scalar('Metrics/Precision', fold_precision, epoch)
        writer.add_scalar('Metrics/Recall', fold_recall, epoch)
        writer.add_scalar('Metrics/F1_Score', fold_f1, epoch)
        writer.add_scalar('Metrics/MCC', fold_mcc, epoch)
        writer.add_scalar('Metrics/ROC_AUC', fold_roc_auc, epoch)
        writer.add_scalar('Metrics/Log_Loss', fold_log_loss, epoch)

        print(f'Эпоха {epoch + 1}/{num_epochs}, Потеря на тесте: {test_loss / len(test_loader):.4f}, Точность: {fold_accuracy:.4f}, Точность предсказания: {fold_precision:.4f}, Полнота: {fold_recall:.4f}, F1-мера: {fold_f1:.4f}, MCC: {fold_mcc:.4f}, ROC AUC: {fold_roc_auc:.4f}, Логарифмическая потеря: {fold_log_loss:.4f}')

        if fold_f1 > best_f1_score:
            best_f1_score = fold_f1
            torch.save(model.state_dict(), f'model_fold_{fold}_best.pth')

        scheduler.step(test_loss / len(test_loader))

    writer.close()
    fold += 1

print('Кросс-валидация завершена.')