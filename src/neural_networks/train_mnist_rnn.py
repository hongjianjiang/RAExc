import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

# ------------------------------
# Dataset for [label, x1, x2, ..., x28] CSV
# ------------------------------

class SequenceVectorDataset(Dataset):
    """
    Dataset for CSVs where the first column is the label and the next 28 columns are the input vector.
    Input CSV format: [label, x1, x2, ..., x28] (29 columns, no header)
    """
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 29:
                    label = float(line[0])
                    vector = [float(x) for x in line[1:29]]
                    self.data.append(vector)
                    self.labels.append(label)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        print(f"Data shape: {self.data.shape}")  # Should be (N, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns shape (28,)
        return self.data[idx], self.labels[idx]

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

# ------------------------------
# Reproducibility
# ------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Dataset for preprocessed binary strings
# ------------------------------

class MNISTBinaryStringDataset(Dataset):
    """
    Dataset for preprocessed MNIST data with binary strings.
    Input CSV format: [label, row0, row1, ..., row27] (no header)
    Each row is a 28-char binary string like "0000111100001111..."
    """
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 29:  # label + 28 rows
                    label = float(line[0])
                    # Convert each binary string to a list of floats
                    rows = []
                    for i in range(28):
                        row_str = line[i + 1]
                        row_floats = [float(c) for c in row_str]
                        rows.append(row_floats)
                    self.data.append(rows)
                    self.labels.append(label)
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        print(f"Data shape: {self.data.shape}")  # Should be (N, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns shape (28, 28) - 28 rows, each with 28 binary values
        return self.data[idx], self.labels[idx]


# ------------------------------
# Dataset for original MNIST CSV (785 columns: label + 784 pixels)
# ------------------------------

class MNISTOriginalDataset(Dataset):
    """
    Dataset for original MNIST CSV format.
    Input CSV format: [label, pixel0, pixel1, ..., pixel783] (785 columns, no header)
    Pixels are 0-255, normalized to 0-1.
    Binary classification: label 1 -> 1.0, other digits -> 0.0
    """
    def __init__(self, csv_file, target_digit=1):
        self.data = []
        self.labels = []
        self.target_digit = target_digit
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= 785:  # label + 784 pixels
                    original_label = int(line[0])
                    # Binary: is it the target digit?
                    binary_label = 1.0 if original_label == target_digit else 0.0
                    
                    # Parse 784 pixels (0-255), normalize to 0-1
                    pixels = [float(p) / 255.0 for p in line[1:785]]
                    # Reshape to 28x28
                    rows = [pixels[i*28:(i+1)*28] for i in range(28)]
                    
                    self.data.append(rows)
                    self.labels.append(binary_label)
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
        num_pos = int(self.labels.sum().item())
        num_neg = len(self.labels) - num_pos
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        print(f"Data shape: {self.data.shape}")  # Should be (N, 28, 28)
        print(f"Target digit: {target_digit} | Positive: {num_pos} | Negative: {num_neg}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ------------------------------
# LSTM Classifier for Binary Image Sequences
# ------------------------------

class LSTMClassifier(nn.Module):

    def __init__(self, input_size=28, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, 28, 28) - 28 timesteps, 28 features per timestep
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(1)
        return torch.sigmoid(logits)


# ------------------------------
# Evaluation
# ------------------------------

def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            probs = model(x)

            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    TP = np.sum((all_preds == 1) & (all_labels == 1))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    acc = (TP + TN) / len(all_labels)

    return TP, FP, TN, FN, acc


# ------------------------------
# Training
# ------------------------------

def train(train_csv='mnist_train.csv', test_csv='mnist_test.csv', use_original=True):

    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001


    # --- Choose dataset type here ---
    # Uncomment the dataset you want to use:

    # 1. For [label, x1, ..., x28] format:
    train_ds = SequenceVectorDataset(train_csv)
    test_ds = SequenceVectorDataset(test_csv)

    # 2. For original MNIST format (785 columns):
    # train_ds = MNISTOriginalDataset(train_csv, target_digit=1)
    # test_ds = MNISTOriginalDataset(test_csv, target_digit=1)

    # 3. For preprocessed binary string format:
    # train_ds = MNISTBinaryStringDataset(train_csv)
    # test_ds = MNISTBinaryStringDataset(test_csv)


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # For SequenceVectorDataset, input_size=28, but input shape is (batch, 28)
    # LSTM expects (batch, seq_len, input_size), so we unsqueeze(-1) in the training loop
    model = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"Training samples: {len(train_ds)}")


    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # For SequenceVectorDataset, x shape is (batch, 28), reshape to (batch, 28, 1)
            x = x.unsqueeze(-1)
            optimizer.zero_grad()
            probs = model(x)
            loss = criterion(probs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {total_loss/len(train_loader):.4f}")

    # evaluation


    # For evaluation, also unsqueeze(-1) for each batch
    def eval_loader(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x = x.unsqueeze(-1)
                probs = model(x)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        TP = np.sum((all_preds == 1) & (all_labels == 1))
        FP = np.sum((all_preds == 1) & (all_labels == 0))
        TN = np.sum((all_preds == 0) & (all_labels == 0))
        FN = np.sum((all_preds == 0) & (all_labels == 1))
        acc = (TP + TN) / len(all_labels)
        return TP, FP, TN, FN, acc

    TP, FP, TN, FN, acc = eval_loader(test_loader)

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"TP: {TP}  TN: {TN}")
    print(f"FP: {FP}  FN: {FN}")
    print("==============================")

    torch.save(model.state_dict(), "mnist_binary_lstm.pt")

    print("\nModel saved to mnist_binary_lstm.pt")
    
    # Trace with correct shape: (batch=1, seq_len=28, features=28)
    dummy_input = torch.randn(1, 28, 28).to(DEVICE)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save the JIT model to the lstm folder
    import os
    lstm_folder = os.path.join(os.path.dirname(__file__), "..", "..", "nn_models", "lstm")
    jit_path = os.path.join(lstm_folder, "mnist.pt")
    traced_model.save(jit_path)
    print(f"JIT model saved to {jit_path}")
    return model


# ------------------------------
# Prediction Example
# ------------------------------

def predict_sample(model, row_strings):
    """
    Predict on a sequence of 28 binary strings.
    Each string is 28 characters like "0000111100001111..."
    """
    model.eval()

    # Convert strings to float arrays
    rows = []
    for row_str in row_strings:
        row_floats = [float(c) for c in row_str]
        rows.append(row_floats)
    
    # Shape: (1, 28, 28)
    tensor = torch.tensor([rows], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        prob = model(tensor).item()

    pred = 1 if prob >= 0.5 else 0

    print("\nSample prediction")
    print("-----------------")
    print(f"Probability of '1': {prob:.4f}")
    print(f"Prediction: {pred}")
    return pred, prob


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    import os
    
    # Use mnist_train.csv and mnist_test.csv in [label, x1, ..., x28] format
    train_csv = 'mnist_train.csv'
    test_csv = 'mnist_test.csv'

    model = train(train_csv, test_csv)

    # Example prediction: single vector of 28 zeros
    sample_vector = [0.0] * 28
    # predict_sample expects a list of 28 strings, so convert floats to strings if needed
    sample_row_strings = [str(int(x)) * 28 for x in sample_vector]  # for compatibility, but you may want to adapt this
    # If you want to predict on a real vector, use model(torch.tensor([[vector]], dtype=torch.float32).to(DEVICE).unsqueeze(-1))

    # Copy to nn_models/lstm/
    import shutil
    dest = '../../nn_models/lstm/mnist.pt'
    if os.path.exists('mnist.pt'):
        shutil.copy('mnist.pt', dest)
        print(f"Copied mnist.pt to {dest}")