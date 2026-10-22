import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os

# ------------------------------
# Configuration & Device
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------
# Dataset: [Label, F1, F2, ..., F28]
# ------------------------------
class MNISTSequenceDataset(Dataset):
    def __init__(self, csv_file, target_digit=1):
        self.data = []
        self.labels = []
        
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found.")
            return

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if not line or len(line) < 29: continue
                
                # First value is the label
                try:
                    label = int(float(line[0]))
                    binary_label = 1.0 if label == target_digit else 0.0
                    
                    # Next 28 values are features
                    features = np.array(line[1:29], dtype=np.float32) / 255.0
                    
                    # Shape: (28, 1) -> 28 steps, 1 feature per step
                    self.data.append(features.reshape(28, 1))
                    self.labels.append(binary_label)
                except ValueError:
                    continue # Skip header or corrupted rows
        
        if self.data:
            self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
            self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
            print(f"Loaded {len(self.data)} samples from {csv_file}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ------------------------------
# Transformer Model
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        # 1. Expand single pixel value to a vector
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output head
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, 28, 1)
        x = self.embedding(x)      
        x = self.pos_encoder(x)
        x = self.transformer(x)    
        
        # Pooling: Use the mean of all 28 steps
        x = x.mean(dim=1)          
        return torch.sigmoid(self.fc(x)).squeeze(1)

# ------------------------------
# Training and Evaluation
# ------------------------------
def train():
    # Setup
    train_csv = 'mnist_train.csv'
    test_csv = 'mnist_test.csv'
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001

    train_ds = MNISTSequenceDataset(train_csv)
    test_ds = MNISTSequenceDataset(test_csv)
    
    if len(train_ds) == 0:
        print("Dataset is empty. Check your CSV path and format.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TransformerClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCELoss()

    print(f"Training started on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss/len(train_loader):.4f}")

    # Final Accuracy Check
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = (model(x) >= 0.5).float()
            correct += (preds == y).sum().item()
    
    print(f"\nFinal Test Accuracy: {(correct/len(test_ds))*100:.2f}%")

    # --- Robust Saving ---
    # 1. State Dict (Safe for python loading)
    # torch.save(model.state_dict(), "transformer_state.pt")
    
    # Trace with correct shape: (batch=1, seq_len=28, features=1)
    dummy_input = torch.zeros(1, 28, 1).to(DEVICE)

    # Trace the model
    # traced_model = torch.jit.trace(model, dummy_input)

    # Save the JIT model to the transformer folder
    import os
    # transformer_folder = os.path.join(os.path.dirname(__file__), "..", "..", "nn_models", "transformer")
    # jit_path = os.path.join(transformer_folder, "mnist.pt")
    # traced_model.save("mnist.pt")
    # print(f"JIT model saved to mnist.pt")
    return model

def predict_single_vector(model, vector_28):
    """
    Predict on a single list of 28 values (no label).
    """
    model.eval()
    features = np.array(vector_28, dtype=np.float32) / 255.0
    tensor = torch.tensor(features.reshape(1, 28, 1), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        prob = model(tensor).item()
    
    print(f"Probability: {prob:.4f} | Prediction: {1 if prob >= 0.5 else 0}")
    return prob

if __name__ == "__main__":
    trained_model = train()
    
    # Test with a dummy vector
    if trained_model:
        dummy_data = [0]*28
        predict_single_vector(trained_model, dummy_data)