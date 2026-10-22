import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import time
from datetime import timedelta


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

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_len):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        bsz, seq_len, _ = x.size()
        # Use correct padding value for mask detection
        mask = (x.squeeze(-1) == -999.0)  # Use the actual padding value
        x = self.input_proj(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.position_embedding(pos_ids)
        
        # Check if any sequence is entirely padding
        has_valid_tokens = (~mask).any(dim=1)  # [batch_size]
        
        # For sequences that are entirely padding, use a different approach
        if not has_valid_tokens.all():
            # Create a dummy token for entirely padded sequences
            x[mask.all(dim=1)] = 0.0  # Reset to zeros for fully padded sequences
            # Adjust mask to allow at least one token per sequence
            adjusted_mask = mask.clone()
            fully_padded = mask.all(dim=1)
            adjusted_mask[fully_padded, 0] = False  # Unmask first token for fully padded sequences
        else:
            adjusted_mask = mask
        
        encoded = self.encoder(x, src_key_padding_mask=adjusted_mask)
        valid_mask = (~adjusted_mask).unsqueeze(-1).float()
        summed = (encoded * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
        pooled = summed / counts
        logits = self.classifier(pooled).squeeze(-1)
        return logits

# --- Training & Evaluation ---
def train(model, loader, optim, criterion, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)  # Model outputs logits
        loss = criterion(logits, y)
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
            logits = model(x)  # Model outputs logits
            total += criterion(logits, y).item()
            probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
            pred = (probs >= 0.5).float()  # Convert to binary predictions
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
            logits = model(x)  # Model outputs logits
            probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
            pred = (probs >= 0.5).float()  # Convert to binary predictions
            
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
    """Load a saved Transformer model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint (not from args)
    config = checkpoint['model_config']
    model = TransformerClassifier(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_len']
    ).to(device)
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"    Best accuracy: {checkpoint['best_accuracy']:.4f}")
    print(f"    Training stopped at step: {checkpoint['final_step']}")
    print(f"    Model architecture: d_model={config['d_model']}, nhead={config['nhead']}, num_layers={config['num_layers']}")
    
    return model, checkpoint

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transformer Sequence Classifier')
    parser.add_argument('--language', type=str, default='L1',
                        help='Language identifier (e.g., L1, L2, etc.)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pretrained model file (.pt)')
    parser.add_argument('--inference_only', action='store_true',
                        help='Only run inference when loading a model (skip training)')
    parser.add_argument('--max_len', type=int, default=51,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=1,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='Transformer feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
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
    CSV_FILE    = "../dataset/" + LANGUAGE + ".csv"
    MAX_LEN     = args.max_len
    BATCH_SIZE  = args.batch_size
    D_MODEL     = args.d_model
    NHEAD       = args.nhead
    NUM_LAYERS  = args.num_layers
    DIM_FEEDFORWARD = args.dim_feedforward
    DROPOUT     = args.dropout
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
    TEST_CSV_FILE = "../dataset/test_" + LANGUAGE + ".csv"
    test_ds = CSVSequenceDataset(TEST_CSV_FILE, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Check if we should load a pre-trained model or train from scratch
    if LOAD_MODEL_PATH:
        # Load pre-trained model
        model, checkpoint = load_model(LOAD_MODEL_PATH, DEVICE)
        best_acc = checkpoint['best_accuracy']
        
        # Get actual model configuration from loaded model
        loaded_config = checkpoint['model_config']
        actual_d_model = loaded_config['d_model']
        actual_num_layers = loaded_config['num_layers']
        
        if args.inference_only:
            print("Inference-only mode: Skipping training, running inference only.")
        else:
            print(f"Continuing training from pre-trained model (current best: {best_acc:.4f})...")
            print(f"Using loaded model architecture: d_model={actual_d_model}, num_layers={actual_num_layers}")
            
            # Check if command line args differ from loaded model
            if D_MODEL != actual_d_model or NUM_LAYERS != actual_num_layers:
                print(f"Warning: Command line args (d_model={D_MODEL}, num_layers={NUM_LAYERS}) differ from loaded model")
                print(f"Using loaded model architecture instead")
            
            # Initialize AdamW optimizer for continued training
            optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss since model outputs logits
            
            # Add learning rate scheduler for stability
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
    else:
        # Train from scratch
        print("Training from scratch...")
        model = TransformerClassifier(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            max_len=MAX_LEN
        ).to(DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss since model outputs logits
        
        # Add learning rate scheduler for stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        best_acc = 0.0

    # Only run training if not in inference-only mode
    if not (LOAD_MODEL_PATH and args.inference_only):
        print(f"Starting training for maximum {MAX_STEPS} steps...")
        print(f"Using AdamW optimizer with lr={LR}, weight_decay={WEIGHT_DECAY}")
        print(f"Early stopping when validation accuracy reaches {TARGET_ACC*100:.1f}% (3 times)")
        print(f"Evaluating every {EVAL_INTERVAL} steps")
        
        step = 0
        best_model_state = None
        target_acc_count = 0  # Counter for how many times we've hit target accuracy
        train_iterator = iter(train_loader)
        
        # Create tqdm progress bar for training steps
        pbar = tqdm(total=MAX_STEPS, desc="Training Progress")
        
        # Start timing
        training_start_time = time.time()
        
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
            logits = model(x)  # Model outputs logits
            loss = crit(logits, y)
            
            # Check for NaN loss and skip if found
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {step}! Skipping this batch.")
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()
            
            step += 1
            pbar.update(1)  # Update progress bar
            
            # Evaluate periodically
            if step % EVAL_INTERVAL == 0:
                val_metrics = evaluate_detailed(model, val_loader, DEVICE)
                val_acc = val_metrics['Accuracy']
                
                # Update learning rate scheduler
                if 'scheduler' in locals():
                    scheduler.step(val_acc)
                
                # print(f"Step {step:6d}/{MAX_STEPS} - Loss: {loss.item():.4f} | Val Acc: {val_acc:.6f} ({val_acc*100:.4f}%) | "
                #       f"F1: {val_metrics['F1_Score']:.6f} | LR: {optim.param_groups[0]['lr']:.2e}")
                
                # Update progress bar description with current metrics
                pbar.set_description(f"Training - Loss: {loss.item():.4f} | Val Acc: {val_acc*100:.2f}%")
                
                # Early stopping check - need to hit target accuracy 3 times
                if val_acc >= TARGET_ACC:
                    target_acc_count += 1
                    print(f"Target accuracy {TARGET_ACC*100:.1f}% reached! Count: {target_acc_count}/3")
                    
                    if target_acc_count >= 3:
                        print(f"Target accuracy reached 3 times at step {step}! Stopping early.")
                        best_model_state = model.state_dict().copy()
                        pbar.close()  # Close progress bar before breaking
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
        
        # Save the best model (only if we trained)
        model_save_path = f'transformer_best_model_{LANGUAGE}_{best_acc:.4f}.pt'
        if best_model_state is not None:
            torch.save({
                'model_state_dict': best_model_state,
                'best_accuracy': best_acc,
                'final_step': step,
                'model_config': {
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'dim_feedforward': DIM_FEEDFORWARD,
                    'dropout': DROPOUT,
                    'max_len': MAX_LEN
                },
                'training_config': {
                    'max_len': MAX_LEN,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'max_steps': MAX_STEPS,
                    'target_accuracy': TARGET_ACC
                }
            }, model_save_path)
            print(f"Best model saved to: {model_save_path}")
        
        pbar.close()  # Close the progress bar
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        training_time_str = str(timedelta(seconds=int(training_duration)))
        
        print(f"Training completed! Final step: {step}, Best accuracy: {best_acc:.4f}")
        print(f"Total training time: {training_time_str} ({training_duration:.2f} seconds)")
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
        actual_d_model = config['d_model']
        actual_num_layers = config['num_layers']
    else:
        actual_d_model = D_MODEL
        actual_num_layers = NUM_LAYERS
    
    print("Model Configuration:")
    print(f"  d_model:        {actual_d_model}")
    print(f"  nhead:          {NHEAD}")
    print(f"  num_layers:     {actual_num_layers}")
    print(f"  dim_feedforward: {DIM_FEEDFORWARD}")
    print(f"  dropout:        {DROPOUT}")
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
    log_filename = f"transformer_evaluation_{LANGUAGE}.log"
    
    with open(log_filename, 'a') as log_file:  # Changed to append mode
        log_file.write("\n" + "="*60 + "\n")
        log_file.write("COMPREHENSIVE TEST EVALUATION - TRANSFORMER\n")
        log_file.write("="*60 + "\n")
        log_file.write(f"Evaluation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        log_file.write("Model Configuration:\n")
        log_file.write(f"  Language:       {LANGUAGE}\n")
        log_file.write(f"  d_model:        {actual_d_model}\n")
        log_file.write(f"  nhead:          {NHEAD}\n")
        log_file.write(f"  num_layers:     {actual_num_layers}\n")
        log_file.write(f"  dim_feedforward: {DIM_FEEDFORWARD}\n")
        log_file.write(f"  dropout:        {DROPOUT}\n")
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
            logits = model(x)  # Model outputs logits
            prob = torch.sigmoid(logits).item()  # Apply sigmoid to get probability
        return prob >= 0.5 
    
    # count = 0
    # for seq in generate_only_positive_samples(LANGUAGE, num_pos=1000, max_len=MAX_LEN):
    #     if not predict(seq):
    #         count += 1
    # print(f"Predicted {count} out of 1000 generated positive samples as label=0 (false negatives)")
    # print(f"Generated positive sample accuracy: {(1000-count)/1000*100:.2f}%")
