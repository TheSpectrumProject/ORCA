import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import optuna


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, sequence_length, label):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.label = label

        num_columns = self.data.shape[1]
        expected_columns = 8  # Updated to match new input dimensions

        if num_columns != expected_columns:
            raise ValueError(f"CSV file {csv_file} has {num_columns} columns. Expected {expected_columns} columns.")

        self.input_size = num_columns

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.sequence_length, :].values.astype(np.float32)
        target = torch.tensor(self.label, dtype=torch.int64)

        if sequence.shape[1] != self.input_size:
            raise ValueError(f"Mismatch: Expected {self.input_size}, got {sequence.shape[1]}")

        return torch.tensor(sequence), target


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embedding = nn.Linear(input_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        output = self.transformer(x, x)
        output = output[-1, :, :]  # Take the output of the last time step
        output = self.fc(output)
        return output


def objective(trial, train_loader, val_loader, device):
    embed_dim = trial.suggest_categorical('embed_dim', [32, 64, 128])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    model = TimeSeriesTransformer(input_size=8, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                                  dim_feedforward=dim_feedforward, dropout=dropout).to(device)  # Updated input_size
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                epoch_val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader.dataset)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        trial.report(epoch_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    return epoch_val_loss


def train_final_model(train_loader, val_loader, best_params, device):
    model = TimeSeriesTransformer(input_size=8, embed_dim=best_params['embed_dim'], num_heads=best_params['num_heads'],
                                  num_layers=best_params['num_layers'], dim_feedforward=best_params['dim_feedforward'],
                                  dropout=best_params['dropout']).to(device)  # Updated input_size
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    epochs = 50
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print("Epoch " + str(epoch) + " finished.")

    torch.save(model.state_dict(), 'best_model.pth')
    print("Final model trained and saved.")

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.view_as(pred).cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()  # Close plot to avoid blocking


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_A_path = './dataset_A'
    dataset_B_path = './dataset_B'

    all_files = []
    for folder_path, label in [(dataset_A_path, 0), (dataset_B_path, 1)]:
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
        for file in files:
            all_files.append((file, label))

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    sequence_length = 10  # Define the sequence length for time series

    train_datasets = [TimeSeriesDataset(file, sequence_length, label) for file, label in train_files]
    val_datasets = [TimeSeriesDataset(file, sequence_length, label) for file, label in val_files]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_datasets), batch_size=32,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size=32, shuffle=False)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=20)

    best_params = study.best_params
    print(f'Best Parameters: {best_params}')

    model = train_final_model(train_loader, val_loader, best_params, device)

    test_files = [file for file, _ in val_files]  # Reuse validation files for testing
    test_dataset = torch.utils.data.ConcatDataset(
        [TimeSeriesDataset(file, sequence_length, label) for file, label in val_files])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()
