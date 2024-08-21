import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader

# Dataset class for prediction
class PredictionDataset(Dataset):
    def __init__(self, csv_file, sequence_length, scaler):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        # Standardize the data using the provided scaler
        self.data.iloc[:, :] = scaler.transform(self.data)

    def __len__(self):
        # Ensuring the length accounts for the sliding window
        return max(1, len(self.data) - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Extract the sliding window sequence
        sequence = self.data.iloc[idx:idx + self.sequence_length, :].values.astype(np.float32)

        # Debugging output to trace data during the process
        print(f"Index: {idx}, Sequence:\n{sequence}")

        return torch.tensor(sequence)


# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape (1, max_len, embed_dim)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

# Transformer model class
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
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
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        output = self.transformer(x, x)
        output = output[-1, :, :]  # Take the output of the last time step
        output = self.fc(output)
        return output

# Prediction function
def predict(csv_file, model_path, scaler_path, sequence_length, device):
    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Load the model with the specified parameters
    model = TimeSeriesTransformer(
        input_size=8,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1378573652063344
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create dataset and dataloader for prediction
    prediction_dataset = PredictionDataset(csv_file, sequence_length, scaler)
    prediction_loader = DataLoader(prediction_dataset, batch_size=1, shuffle=False)

    probabilities = []
    with torch.no_grad():
        for idx, data in enumerate(prediction_loader):
            start_idx = idx  # Adjust this based on how DataLoader fetches your data
            print(f"Processing batch {idx} (starting from data index {start_idx})", flush=True)
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)  # Apply softmax to get probabilities
            prob_class_1 = probs[:, 1].item()  # Extract probability for class 1
            probabilities.append(prob_class_1)

    return probabilities

# Main function for making predictions
def main(csv_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #csv_file = './dataset_A/random_data_2.csv'  # Replace with your new data CSV file
    model_path = 'best_model.pth'  # Replace with your trained model path
    scaler_path = 'scaler.pkl'  # Replace with your scaler path
    sequence_length = 10  # The same sequence length used during training

    probabilities = predict(csv_file, model_path, scaler_path, sequence_length, device)
    print(f"Class 1 Probabilities: {probabilities}")
    average = sum(probabilities) / len(probabilities)
    print("Average:",average)
    return average

if __name__ == "__main__":
    main("./dataset_A/random_data_2.csv")
