import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt



# --- Reproducibility ---
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# For deterministic behavior (may slow down training)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class CSVSequenceDataset(Dataset):
    """Dataset for loading and normalizing sequences and labels from CSV file"""
    def __init__(self, csv_file: str, max_len: int):
        self.csv_file = csv_file
        self.max_len = max_len
        self.pad_value = -999.0  # Normalized padding value (mean = 0)
        self.data = self._load_data()

    def _load_data(self):
        """Load data from CSV file and normalize each sequence"""
        import csv
        data = []
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) > 1:  # Ensure we have at least sequence + label
                    sequence = [float(x) for x in line[:-1]]
                    label = float(line[-1])

                    # Normalize using z-score: (x - mean) / std
                    mean = np.mean(sequence)
                    std = np.std(sequence)
                    if std == 0:
                        # Avoid division by zero; use zeros
                        norm_seq = [0.0 for _ in sequence]
                    else:
                        norm_seq = [(x - mean) / std for x in sequence]

                    data.append((norm_seq, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]

        # Truncate or pad
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        elif len(seq) < self.max_len:
            seq = seq + [self.pad_value] * (self.max_len - len(seq))

        return torch.tensor(seq, dtype=torch.float), torch.tensor(label, dtype=torch.float)

# Collate for DataLoader
def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

# --- LSTM Classifier for Rational Sequences ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        # No embedding layer needed - work directly with rational numbers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len) for sequences of rational numbers
        # Reshape to (batch_size, seq_len, 1) for LSTM input
        x = x.unsqueeze(-1)  # Add feature dimension
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # Use last time step
        logits = self.fc(last).squeeze(1)
        return torch.sigmoid(logits)

# --- Training & Evaluation ---
def train(model, loader, optim, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        probs = model(x)  # Already contains sigmoid output
        loss = criterion(probs, y)
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)  # Already contains sigmoid output
            total += criterion(probs, y).item()
            pred = (probs >= 0.5).float()  # No need for sigmoid since model outputs probabilities
            correct += (pred == y).sum().item()
    return total / len(loader), correct / len(loader.dataset)

def evaluate_detailed(model, loader, device):
    """
    Comprehensive evaluation with confusion matrix metrics
    Returns: TP, FP, TN, FN, Accuracy, Precision, Recall, F1-Score
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)  # Already contains sigmoid output
            pred = (probs >= 0.5).float()  # No need for sigmoid since model outputs probabilities
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix components
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': int(TP),
        'FP': int(FP), 
        'TN': int(TN),
        'FN': int(FN),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Total_Samples': len(all_labels)
    }

def load_model(model_path: str, device):
    """Load a saved LSTM model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint (not from args)
    config = checkpoint['model_config']
    model = LSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"    Best accuracy: {checkpoint['best_accuracy']:.4f}")
    print(f"    Training stopped at step: {checkpoint['final_step']}")
    print(f"    Model architecture: input_size={config['input_size']}, hidden_size={config['hidden_size']}, num_layers={config['num_layers']}")
    
    return model, checkpoint

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LSTM Sequence Classifier')
    parser.add_argument('--language', type=str, default='LnA',
                        help='Language identifier (e.g., L1, L2, etc.)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pretrained model file (.pt)')
    parser.add_argument('--inference_only', action='store_true',
                        help='Only run inference when loading a model (skip training)')
    parser.add_argument('--train_csv', type=str, default="dataset/LnA_train.csv",
                        help='Explicit training CSV path. Overrides --language-based lookup.')
    parser.add_argument('--test_csv', type=str, default="dataset/LnA_test.csv",
                        help='Explicit test CSV path. Overrides --language-based lookup.')
    parser.add_argument('--max_len', type=int, default=51,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum training steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay rate for AdamW optimizer')
    parser.add_argument('--target_acc', type=float, default=1.0,
                        help='Target accuracy for early stopping')
    parser.add_argument('--practical_perfect', action='store_true',
                        help='Use 99.95%% as "perfect" accuracy (0.9995) instead of exact 1.0')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluate every N steps')
    
    args = parser.parse_args()
    
    # Use arguments
    LANGUAGE    = args.language
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    CSV_FILE    = args.train_csv or str(dataset_dir / f"{LANGUAGE}.csv")
    MAX_LEN     = args.max_len
    BATCH_SIZE  = args.batch_size
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS  = args.num_layers
    MAX_STEPS   = args.max_steps
    LR          = args.lr
    WEIGHT_DECAY = args.weight_decay
    TARGET_ACC  = args.target_acc
    EVAL_INTERVAL = args.eval_interval
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Adjust target accuracy if practical perfect is requested
    if args.practical_perfect:
        TARGET_ACC = 0.9995
        print(f"Using practical perfect accuracy: {TARGET_ACC*100:.2f}%")
    
    # Set pretrained model path from arguments
    LOAD_MODEL_PATH = args.load_model

    # Load training/validation dataset from CSV
    ds = CSVSequenceDataset(CSV_FILE, MAX_LEN)
    train_len = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds)-train_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Load test dataset from CSV
    TEST_CSV_FILE = args.test_csv or str(dataset_dir / f"test_{LANGUAGE}.csv")
    test_ds = CSVSequenceDataset(TEST_CSV_FILE, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Check if we should load a pre-trained model or train from scratch
    if LOAD_MODEL_PATH:
        # Load pre-trained model
        model, checkpoint = load_model(LOAD_MODEL_PATH, DEVICE)
        best_acc = checkpoint['best_accuracy']
        
        # Get actual model configuration from loaded model
        loaded_config = checkpoint['model_config']
        actual_hidden_size = loaded_config['hidden_size']
        actual_num_layers = loaded_config['num_layers']
        
        if args.inference_only:
            print("Inference-only mode: Skipping training, running inference only.")
        else:
            print(f"Continuing training from pre-trained model (current best: {best_acc:.4f})...")
            print(f"Using loaded model architecture: hidden_size={actual_hidden_size}, num_layers={actual_num_layers}")
            
            # Check if command line args differ from loaded model
            if HIDDEN_SIZE != actual_hidden_size or NUM_LAYERS != actual_num_layers:
                print(f"Warning: Command line args (hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}) differ from loaded model")
                print(f"Using loaded model architecture instead")
            
            # Initialize AdamW optimizer for continued training
            optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.BCELoss()  # Use BCELoss since sigmoid is applied in forward pass
    else:
        # Train from scratch
        print("Training from scratch...")
        # Model now takes input_size=1 (single rational number per timestep)
        model = LSTMClassifier(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.BCELoss()  # Use BCELoss since sigmoid is applied in forward pass
        best_acc = 0.0

    # Only run training if not in inference-only mode
    if not (LOAD_MODEL_PATH and args.inference_only):
        print(f"Starting training for maximum {MAX_STEPS} steps...")
        print(f"Using AdamW optimizer with lr={LR}, weight_decay={WEIGHT_DECAY}")
        print(f"Early stopping when validation accuracy reaches {TARGET_ACC*100:.1f}% (3 times)")
        print(f"Evaluating every {EVAL_INTERVAL} steps")

        import time
        train_start_time = time.time()

        step = 0
        best_model_state = None
        target_acc_count = 0  # Counter for how many times we've hit target accuracy
        train_iterator = iter(train_loader)
        steps_list = []
        acc_list = []
        latex_steps = []
        latex_accs = []

        while step < MAX_STEPS:
            model.train()
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x, y = next(train_iterator)

            # Training step
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            probs = model(x)  # Already contains sigmoid output
            loss = crit(probs, y)
            loss.backward()
            optim.step()

            step += 1

            # Evaluate periodically

            if step % EVAL_INTERVAL == 0:
                val_loss, val_acc = evaluate(model, val_loader, crit, DEVICE)
                print(f"Step {step:6d}/{MAX_STEPS} - Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                steps_list.append(step)
                acc_list.append(val_acc)

                # Output for LaTeX every 2000 steps
                if step % 2000 == 0:
                    latex_steps.append(step)
                    latex_accs.append(val_acc)
                    print(f"[LATEX] Step: {step}, Accuracy: {val_acc:.6f}")

                # Early stopping check - need to hit target accuracy 3 times
                if val_acc >= TARGET_ACC:
                    target_acc_count += 1
                    print(f"Target accuracy {TARGET_ACC*100:.1f}% reached! Count: {target_acc_count}/3")

                    if target_acc_count >= 3:
                        print(f"Target accuracy reached 3 times at step {step}! Stopping early.")
                        best_model_state = model.state_dict().copy()
                        break
                else:
                    if target_acc_count > 0:
                        print(f"    Accuracy dropped below target. Resetting counter (was {target_acc_count})")
                        target_acc_count = 0

                # Track best accuracy and save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    print(f"    New best validation accuracy: {best_acc:.6f} ({best_acc*100:.4f}%)")

                    if val_acc >= TARGET_ACC - 0.0001:
                        print(f"    Very close to target! Current: {val_acc:.6f}, Target: {TARGET_ACC:.6f}")

        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time

        # Save the best model (only if we trained)
        model_save_path = f'lstm_best_model_{LANGUAGE}_{best_acc:.4f}.pt'
        if best_model_state is not None:
            torch.save({
                'model_state_dict': best_model_state,
                'best_accuracy': best_acc,
                'final_step': step,
                'model_config': {
                    'input_size': 1,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS
                },
                'training_config': {
                    'max_len': MAX_LEN,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LR,
                    'max_steps': MAX_STEPS,
                    'target_accuracy': TARGET_ACC
                }
            }, model_save_path)
            print(f"Best model saved to: {model_save_path}")

        print(f"Training completed! Final step: {step}, Best accuracy: {best_acc:.4f}")
        print(f"Total training time: {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)")

        # Plot accuracy vs step
        if len(steps_list) > 0:
            plt.figure()
            plt.plot(steps_list, acc_list, marker='o')
            plt.xlabel('Step')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy vs Training Step')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f'lstm_val_acc_vs_step_{LANGUAGE}.png')
            plt.show()

            # Output LaTeX table for accuracy every 2000 steps
            if len(latex_steps) > 0:
                print("\nLaTeX Table Data (Step, Accuracy):")
                print("\\begin{tabular}{|c|c|}")
                print("\\hline Step & Accuracy \\")
                print("\\hline")
                for s, a in zip(latex_steps, latex_accs):
                    print(f"{s} & {a:.6f} \\")
                print("\\hline")
                print("\\end{tabular}")
    else:
        print("Running inference with loaded model...")

    # Evaluate on test dataset with detailed metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST EVALUATION")
    print("="*60)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get actual model configuration (either from loaded model or training config)
    if LOAD_MODEL_PATH:
        config = checkpoint['model_config']
        actual_hidden_size = config['hidden_size']
        actual_num_layers = config['num_layers']
    else:
        actual_hidden_size = HIDDEN_SIZE
        actual_num_layers = NUM_LAYERS
    
    print("Model Configuration:")
    print(f"  Input Size:     {1}")
    print(f"  Hidden Size:    {actual_hidden_size}")
    print(f"  Number Layers:  {actual_num_layers}")
    print(f"  Max Length:     {MAX_LEN}")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    test_metrics = evaluate_detailed(model, test_loader, DEVICE)
    
    print(f"\nDataset: {TEST_CSV_FILE}")
    print(f"Total test samples: {test_metrics['Total_Samples']}")
    print("\nConfusion Matrix:")
    print(f"  True Positives  (TP): {test_metrics['TP']:>6d}")
    print(f"  False Positives (FP): {test_metrics['FP']:>6d}")
    print(f"  True Negatives  (TN): {test_metrics['TN']:>6d}")
    print(f"  False Negatives (FN): {test_metrics['FN']:>6d}")
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.6f} ({test_metrics['Accuracy']*100:.4f}%)")
    print(f"  Precision: {test_metrics['Precision']:.6f} ({test_metrics['Precision']*100:.4f}%)")
    print(f"  Recall:    {test_metrics['Recall']:.6f} ({test_metrics['Recall']*100:.4f}%)")
    print(f"  F1-Score:  {test_metrics['F1_Score']:.6f} ({test_metrics['F1_Score']*100:.4f}%)")
    
    # Also evaluate on validation set for comparison
    print("\nValidation Set Comparison:")
    val_metrics = evaluate_detailed(model, val_loader, DEVICE)
    print(f"  Val Accuracy:  {val_metrics['Accuracy']:.6f} ({val_metrics['Accuracy']*100:.4f}%)")
    print(f"  Val Precision: {val_metrics['Precision']:.6f} ({val_metrics['Precision']*100:.4f}%)")
    print(f"  Val Recall:    {val_metrics['Recall']:.6f} ({val_metrics['Recall']*100:.4f}%)")
    print(f"  Val F1-Score:  {val_metrics['F1_Score']:.6f} ({val_metrics['F1_Score']*100:.4f}%)")
    print("="*60)
    
    # Save evaluation results to log file
    import datetime
    log_filename = f"lstm_evaluation_{LANGUAGE}.log"
    
    with open(log_filename, 'a') as log_file:  # Changed to append mode
        log_file.write("\n" + "="*60 + "\n")
        log_file.write("COMPREHENSIVE TEST EVALUATION\n")
        log_file.write("="*60 + "\n")
        log_file.write(f"Evaluation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        log_file.write("Model Configuration:\n")
        log_file.write(f"  Language:       {LANGUAGE}\n")
        log_file.write(f"  Input Size:     {1}\n")
        log_file.write(f"  Hidden Size:    {HIDDEN_SIZE}\n")
        log_file.write(f"  Number Layers:  {NUM_LAYERS}\n")
        log_file.write(f"  Max Length:     {MAX_LEN}\n")
        log_file.write(f"  Batch Size:     {BATCH_SIZE}\n")
        log_file.write(f"  Learning Rate:  {LR}\n")
        log_file.write(f"  Weight Decay:   {WEIGHT_DECAY}\n")
        log_file.write(f"  Total Parameters:     {total_params:,}\n")
        log_file.write(f"  Trainable Parameters: {trainable_params:,}\n")
        log_file.write(f"  Device:         {DEVICE}\n\n")
        
        log_file.write("Dataset Information:\n")
        log_file.write(f"  Test Dataset:   {TEST_CSV_FILE}\n")
        log_file.write(f"  Total test samples: {test_metrics['Total_Samples']}\n\n")
        
        log_file.write("Test Set - Confusion Matrix:\n")
        log_file.write(f"  True Positives  (TP): {test_metrics['TP']:>6d}\n")
        log_file.write(f"  False Positives (FP): {test_metrics['FP']:>6d}\n")
        log_file.write(f"  True Negatives  (TN): {test_metrics['TN']:>6d}\n")
        log_file.write(f"  False Negatives (FN): {test_metrics['FN']:>6d}\n\n")
        
        log_file.write("Test Set - Performance Metrics:\n")
        log_file.write(f"  Accuracy:  {test_metrics['Accuracy']:.6f} ({test_metrics['Accuracy']*100:.4f}%)\n")
        log_file.write(f"  Precision: {test_metrics['Precision']:.6f} ({test_metrics['Precision']*100:.4f}%)\n")
        log_file.write(f"  Recall:    {test_metrics['Recall']:.6f} ({test_metrics['Recall']*100:.4f}%)\n")
        log_file.write(f"  F1-Score:  {test_metrics['F1_Score']:.6f} ({test_metrics['F1_Score']*100:.4f}%)\n\n")
        
        log_file.write("Validation Set - Performance Metrics:\n")
        log_file.write(f"  Val Accuracy:  {val_metrics['Accuracy']:.6f} ({val_metrics['Accuracy']*100:.4f}%)\n")
        log_file.write(f"  Val Precision: {val_metrics['Precision']:.6f} ({val_metrics['Precision']*100:.4f}%)\n")
        log_file.write(f"  Val Recall:    {val_metrics['Recall']:.6f} ({val_metrics['Recall']*100:.4f}%)\n")
        log_file.write(f"  Val F1-Score:  {val_metrics['F1_Score']:.6f} ({val_metrics['F1_Score']*100:.4f}%)\n\n")
        
        log_file.write("="*60 + "\n")
    
    print(f"Evaluation results saved to: {log_filename}")
    print("="*60)

    # Additional inference test on generated positive samples
    print("\nAdditional Inference Test:")
    # inference for sequences loaded from CSV
    def predict(seq):
        model.eval()
        mean = np.mean(seq)
        std = np.std(seq)
        if std == 0:
            norm_seq = [0.0 for _ in seq]
        else:
            norm_seq = [(x - mean) / std for x in seq]
        
        padded_seq = norm_seq + [ds.pad_value] * (MAX_LEN - len(norm_seq))
        x = torch.tensor([padded_seq], dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            prob = model(x).item()  # Model already outputs probabilities with sigmoid
        return prob >= 0.5 
    
    # count = 0
    # for seq in generate_only_positive_samples(LANGUAGE, num_pos=1000, max_len=MAX_LEN):
    #     if not predict(seq):
    #         count += 1
    # print(f"Predicted {count} out of 1000 generated positive samples as label=0 (false negatives)")
    # print(f"Generated positive sample accuracy: {(1000-count)/1000*100:.2f}%")
