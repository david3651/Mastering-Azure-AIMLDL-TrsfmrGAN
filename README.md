```python
## tensorflow API

import argparse

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import SGD

from azureml.core import Run

from keras_azure_ml_cb import AzureMlKerasCallback

# Setup Run

# ---------------------------------------

# Load the current run and ws

run = Run.get_context()

ws = run.experiment.workspace

# Parse parameters

# ---------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--in-train", type=str)

parser.add_argument("--in-test", type=str)

parser.add_argument('--batch-size', type=int, dest='batch_size', default=50)

parser.add_argument('--epochs', type=int, dest='epochs', default=10)

parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100)

parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100)

parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01)

parser.add_argument('--momentum', type=float, dest='momentum', default=0.9)

args = parser.parse_args()

# Load train/test data

# ---------------------------------------

df_train = pd.read_csv(args.in_train)

df_test = pd.read_csv(args.in_test)

y_train = df_train.pop("target").values

X_train = df_train.values

y_test = df_test.pop("target").values

X_test = df_test.values

# Build model

# ---------------------------------------

model = Sequential()

model.add(Dense(args.n_hidden_1, activation='relu', input_dim=X_train.shape[1], kernel_initializer='uniform'))

model.add(Dropout(0.50))

model.add(Dense(args.n_hidden_2, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

sgd = SGD(learning_rate=args.learning_rate, momentum=args.momentum)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Train model

# ---------------------------------------

# Create an Azure Machine Learning monitor callback

azureml_cb = AzureMlKerasCallback(run)

model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, callbacks=[azureml_cb])

# Evaluate model

# ---------------------------------------

scores = model.evaluate(X_test, y_test, batch_size=30)

run.log(model.metrics_names[0], float(scores[0]))

run.log(model.metrics_names[1], float(scores[1]))
```


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```python
## Pytorch API

import argparse

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from azureml.core import Run

from azureml.core.run import _OfflineRun

# Define the neural network model

class Net(nn.Module):

    def __init__(self, input_dim, n_hidden_1, n_hidden_2):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, n_hidden_1)

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)

        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(n_hidden_2, 1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))

        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))

        x = self.dropout2(x)

        x = torch.sigmoid(self.fc3(x))

        return x

# Setup Run

# ---------------------------------------

# Load the current run and ws

run = Run.get_context()

ws = run.experiment.workspace if not isinstance(run, _OfflineRun) else None

# Parse parameters

# ---------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--in-train", type=str)

parser.add_argument("--in-test", type=str)

parser.add_argument('--batch-size', type=int, dest='batch_size', default=50)

parser.add_argument('--epochs', type=int, dest='epochs', default=10)

parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100)

parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100)

parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01)

parser.add_argument('--momentum', type=float, dest='momentum', default=0.9)

args = parser.parse_args()

# Load train/test data

# ---------------------------------------

df_train = pd.read_csv(args.in_train)

df_test = pd.read_csv(args.in_test)

y_train = df_train.pop("target").values

X_train = df_train.values

y_test = df_test.pop("target").values

X_test = df_test.values

# Convert data to PyTorch tensors

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Build model

# ---------------------------------------

model = Net(input_dim=X_train.shape[1], n_hidden_1=args.n_hidden_1, n_hidden_2=args.n_hidden_2)

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

# Train model

# ---------------------------------------

for epoch in range(args.epochs):

    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate model

# ---------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(X_test_tensor)

    loss = criterion(outputs, y_test_tensor)

    accuracy = ((outputs > 0.5) == y_test_tensor).float().mean().item()

run.log("loss", float(loss.item()))

run.log("accuracy", float(accuracy))
```
