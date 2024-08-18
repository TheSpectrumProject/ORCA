import torch
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


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

# Adjusted parameters to match the saved model's parameters
model_params = {
    'input_size': 7,
    'embed_dim': 128,  # Adjust this to match the saved model
    'num_heads': 4,
    'num_layers': 2,
    'dim_feedforward': 256,  # Adjust this if needed
    'dropout': 0.1
}



def load_model(model_path, device):
    model_params = {
        'input_size': 7,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1
    }

    model = TimeSeriesTransformer(**model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_csv(model, csv_file, device):
    data = pd.read_csv(csv_file)
    sequence_length = 10
    num_columns = data.shape[1]

    if num_columns != 7:
        raise ValueError(f"CSV file {csv_file} has {num_columns} columns. Expected 7 columns.")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])

    sequences = torch.tensor(sequences, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(sequences)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1).cpu().numpy()
        probabilities = probabilities.cpu().numpy()

    return predictions, probabilities


def open_model_file():
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    if file_path:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, file_path)


def open_csv_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        csv_path_entry.delete(0, tk.END)
        csv_path_entry.insert(0, file_path)


def run_prediction():
    model_path = model_path_entry.get()
    csv_path = csv_path_entry.get()

    result_area.config(state=tk.NORMAL)
    result_area.delete(1.0, tk.END)  # Clear previous output

    if not model_path or not csv_path:
        result_area.insert(tk.END, "Error: Please provide both model file and CSV file.\n")
        result_area.config(state=tk.DISABLED)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = load_model(model_path, device)
        predictions, probabilities = predict_csv(model, csv_path, device)

        result_text = "Predictions:\n" + "\n".join(
            [f"Sequence {i + 1}: Class {pred}, Probabilities: {prob}" for i, (pred, prob) in
             enumerate(zip(predictions, probabilities))])
        result_area.insert(tk.END, result_text + "\n")

    except Exception as e:
        result_area.insert(tk.END, f"Error: {str(e)}\n")

    result_area.config(state=tk.DISABLED)


# Create the GUI window
window = tk.Tk()
window.title("Time Series Prediction")

tk.Label(window, text="Model File:").grid(row=0, column=0, padx=10, pady=10)
model_path_entry = tk.Entry(window, width=50)
model_path_entry.grid(row=0, column=1, padx=10, pady=10)
tk.Button(window, text="Browse", command=open_model_file).grid(row=0, column=2, padx=10, pady=10)

tk.Label(window, text="CSV File:").grid(row=1, column=0, padx=10, pady=10)
csv_path_entry = tk.Entry(window, width=50)
csv_path_entry.grid(row=1, column=1, padx=10, pady=10)
tk.Button(window, text="Browse", command=open_csv_file).grid(row=1, column=2, padx=10, pady=10)

tk.Button(window, text="Run Prediction", command=run_prediction).grid(row=2, column=0, columnspan=3, padx=10, pady=10)

result_area = scrolledtext.ScrolledText(window, width=80, height=20, state=tk.DISABLED)
result_area.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

window.mainloop()
