import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox

# Define the Transformer model and other components
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

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
        self.softmax = nn.Softmax(dim=1)  # Softmax to get probabilities

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        output = self.transformer(x, x)
        output = output[-1, :, :]  # Take the output of the last time step
        output = self.fc(output)
        output = self.softmax(output)  # Apply softmax to get probabilities
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape (1, max_len, embed_dim)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_model(model_path, input_size, embed_dim, num_heads, num_layers, dim_feedforward, dropout, device):
    model = TimeSeriesTransformer(
        input_size=input_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

def preprocess_data(file_path):
    data = pd.read_csv(file_path).values.astype(np.float32)
    dataset = TimeSeriesDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False)

def predict(model, data_loader, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            probabilities.append(output.cpu().numpy())
    return np.concatenate(probabilities, axis=0)

def prediction2result(probabilities):
    # Extract probability of 'Cheating' (class 0)
    cheating_prob = probabilities[0, 0]  # Assumes single sample; adjust if batch_size > 1
    return cheating_prob

# Define the GUI
class TimeSeriesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Prediction")

        self.model_path = None
        self.file_path = None

        self.setup_gui()

    def setup_gui(self):
        # Create a frame for padding
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Model file selection
        tk.Label(frame, text="Select Model File:", font=("Helvetica", 12)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Button(frame, text="Browse", command=self.load_model_file, width=15).grid(row=0, column=1, padx=5, pady=5)

        # Data file selection
        tk.Label(frame, text="Select Data File:", font=("Helvetica", 12)).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Button(frame, text="Browse", command=self.load_data_file, width=15).grid(row=1, column=1, padx=5, pady=5)

        # Model parameters
        tk.Label(frame, text="Embedding Dimension:", font=("Helvetica", 12)).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.embed_dim_entry = tk.Entry(frame, width=10)
        self.embed_dim_entry.insert(0, "32")  # Default value
        self.embed_dim_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(frame, text="Number of Heads:", font=("Helvetica", 12)).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.num_heads_entry = tk.Entry(frame, width=10)
        self.num_heads_entry.insert(0, "2")  # Default value
        self.num_heads_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(frame, text="Number of Layers:", font=("Helvetica", 12)).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.num_layers_entry = tk.Entry(frame, width=10)
        self.num_layers_entry.insert(0, "3")  # Default value
        self.num_layers_entry.grid(row=4, column=1, padx=5, pady=5)

        tk.Label(frame, text="Feedforward Dim:", font=("Helvetica", 12)).grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.dim_feedforward_entry = tk.Entry(frame, width=10)
        self.dim_feedforward_entry.insert(0, "64")  # Default value
        self.dim_feedforward_entry.grid(row=5, column=1, padx=5, pady=5)

        tk.Label(frame, text="Dropout Rate:", font=("Helvetica", 12)).grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.dropout_entry = tk.Entry(frame, width=10)
        self.dropout_entry.insert(0, "0.1561791975716901")  # Default value
        self.dropout_entry.grid(row=6, column=1, padx=5, pady=5)

        # Predict button
        tk.Button(frame, text="Predict", command=self.make_prediction, width=15, bg="#4CAF50", fg="white", font=("Helvetica", 12)).grid(row=7, column=0, columnspan=2, pady=20)

        # Output display
        self.result_label = tk.Label(frame, text="", font=("Helvetica", 16))
        self.result_label.grid(row=8, column=0, columnspan=2, pady=20)

    def load_model_file(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pth")])
        if self.model_path:
            messagebox.showinfo("Model File", f"Model file selected: {self.model_path}")

    def load_data_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            messagebox.showinfo("Data File", f"Data file selected: {self.file_path}")

    def make_prediction(self):
        if not self.model_path or not self.file_path:
            messagebox.showerror("Error", "Please select both model file and data file.")
            return

        try:
            # Get user-defined model parameters
            embed_dim = int(self.embed_dim_entry.get())
            num_heads = int(self.num_heads_entry.get())
            num_layers = int(self.num_layers_entry.get())
            dim_feedforward = int(self.dim_feedforward_entry.get())
            dropout = float(self.dropout_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid model parameters. Please enter valid numbers.")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = load_model(self.model_path, input_size=8, embed_dim=embed_dim, num_heads=num_heads,
                           num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        data_loader = preprocess_data(self.file_path)
        probabilities = predict(model, data_loader, device)

        cheating_prob = prediction2result(probabilities)
        self.result_label.config(text=f"Probability of Cheating: {cheating_prob:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesApp(root)
    root.mainloop()
