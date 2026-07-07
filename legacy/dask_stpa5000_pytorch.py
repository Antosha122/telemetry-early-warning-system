import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from dask.distributed import Client
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
import time
import tempfile

start_time = time.time()
tempfile.tempdir = 'F:/tempp/'

client = Client(processes=False, dashboard_address=':8794')

def check_emergency_in_3_hours(batch_time, emergency_dates):
    batch_datetime = datetime.strptime(batch_time, '%Y-%m-%d %H:%M:%S')
    future_datetime = batch_datetime + timedelta(hours=3)
    return 1 if future_datetime in emergency_dates else 0

print("Loading data from files...")
opers_data = pd.read_csv('F:/ККАЛ/DataSet Газпром/opers.csv')
stpa_data = pd.read_csv('F:/ККАЛ/DataSet Газпром/stpa5000.csv')

print("Filtering emergency events data...")
emergency_dates = opers_data[opers_data['is_emergency']]['date'].tolist()

def process_stpa_data(chunk):
    zero_mask = (chunk.iloc[:, 1:3600] != 0).any(axis=1)
    chunk_filtered = chunk[zero_mask]
    chunk_filtered['emergency_in_3_hours'] = chunk_filtered['batch_time'].apply(
        lambda x: check_emergency_in_3_hours(x, emergency_dates)
    )
    X = chunk_filtered.iloc[:, 1:-1]
    y = chunk_filtered['emergency_in_3_hours']
    return X, y

print("Creating data transformer...")
transformer = FunctionTransformer(process_stpa_data)

class PyTorchWrapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(PyTorchWrapper, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Splitting data into train and test sets...")
train_stpa_data, test_stpa_data = train_test_split(stpa_data, test_size=0.1, random_state=20)

if len(train_stpa_data) == 0:
    raise ValueError("Insufficient data for training. Please adjust the test_size or check your data.")

print("Applying transformer to training data...")
X_train, y_train = transformer.transform(train_stpa_data)

# Ensure all values in X_train are numeric and convertible to float32
X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce')
X_train_numeric = X_train_numeric.dropna()

# Convert X_train_numeric to torch.Tensor
X_train_tensor = torch.tensor(X_train_numeric.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train[X_train_numeric.index].values, dtype=torch.float32)

print("Creating PyTorch model...")
estimator = PyTorchWrapper(input_size=X_train_tensor.shape[1], output_size=1)

print("Defining loss function and optimizer...")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(estimator.parameters(), lr=0.001)

print("Training the model...")
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = estimator(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Model training complete.")

# Save the trained model
torch.save(estimator.state_dict(), 'trained_model.pth')
print("Trained model saved.")

# Load the saved model
loaded_model = PyTorchWrapper(input_size=X_train_tensor.shape[1], output_size=1)
loaded_model.load_state_dict(torch.load('trained_model.pth'))
loaded_model.eval()

end_time = time.time()
execution_time = end_time - start_time

while True:
    try:
        print("Enter data for analysis (comma-separated, without spaces):")
        user_input = input().strip()

        data_list = user_input.split(',')
        batch_time = data_list[0]
        features = list(map(float, data_list[1:]))

        data_dict = {'batch_time': [batch_time], 'feature': features}
        data = pd.DataFrame(data_dict)

        X_new, _ = transformer.transform(data)
        X_new_numeric = X_new.apply(pd.to_numeric, errors='coerce').dropna()
        X_new_tensor = torch.tensor(X_new_numeric.values, dtype=torch.float32)

        predictions = loaded_model(X_new_tensor)

        for prediction in predictions:
            if prediction.item() > 0.5:
                print("Emergency expected in 3 hours!")
            else:
                print("No emergency expected in 3 hours.")

        time.sleep(3600)

    except Exception as e:
        print(f"Error analyzing data: {e}")
