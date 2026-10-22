# Neural Network Sequence Classification: LSTM vs Transformer

This project implements and compares LSTM and Transformer models for sequence classification tasks on formal languages. The models are trained to recognize patterns in various formal language datasets including Tomita grammars and custom language definitions.

## 🎯 Overview

The project focuses on:
- **Sequence Classification**: Binary classification of sequences belonging to formal languages
- **Model Comparison**: Side-by-side evaluation of LSTM vs Transformer architectures
- **Formal Languages**: Training and testing on languages like L1-L7 and S1-S4 
- **Standardized Training**: Both models use identical training procedures for fair comparison

## 📁 Project Structure

```
transformer/
├── README.md                    # This file
├── Makefile                    # Build and utility commands
├── dataset/                    # Training and test datasets
│   ├── L1.csv, L2.csv, ...    # Training data for languages L1-L7
│   ├── S1.csv, S2.csv, ...    # Training data for languages S1-S4
│   └── test_*.csv             # Test datasets
├── lstm/                      # LSTM implementation
│   └── lstm.py               # LSTM sequence classifier
├── transformer/              # Transformer implementation
│   ├── transformer_encoder.py # Transformer sequence classifier
│   ├── transformer_decoder.py # Transformer decoder variant
│   └── gen_dataset.py         # Dataset generation utilities
└── gen_dataset.py            # Main dataset generation script
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy
```

### Training Models

**Train LSTM on language L1:**
```bash
python3 lstm/lstm.py --language L1 --max_steps 30000
```

**Train Transformer on language L1:**
```bash
python3 transformer/transformer_encoder.py --language L1 --max_steps 30000
```

### Model Evaluation

Both models automatically evaluate on test datasets after training and provide comprehensive metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TP, FP, TN, FN)
- Detailed performance logs

## 🔧 Command Line Options

### Common Parameters (Both LSTM & Transformer)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--language` | Required | Language to train on (L1, L2, ..., S1, S2, ...) |
| `--max_steps` | 30000 | Maximum training steps |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight_decay` | 0.01 | Weight decay for AdamW optimizer |
| `--max_len` | 51 | Maximum sequence length |
| `--target_acc` | 1.0 | Target accuracy for early stopping |
| `--eval_interval` | 100 | Evaluate every N steps |

### LSTM-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden_size` | 128 | LSTM hidden dimension |
| `--num_layers` | 1 | Number of LSTM layers |
| `--dropout` | 0.0 | Dropout rate |

### Transformer-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d_model` | 128 | Transformer model dimension |
| `--nhead` | 1 | Number of attention heads |
| `--num_layers` | 1 | Number of transformer layers |
| `--dim_feedforward` | 256 | Feedforward dimension |

### Advanced Options

| Parameter | Description |
|-----------|-------------|
| `--load_model` | Path to pretrained model (.pt file) |
| `--inference_only` | Skip training, only run inference |
| `--practical_perfect` | Use 99.95% as "perfect" instead of 100% |

## 📊 Datasets

The project includes datasets for various formal languages:

### Tomita Languages (L1-L7)
- **L1**: Sequences over {0,1}* 
- **L2**: Sequences over {0,1}* ending in specific patterns
- **L3-L7**: Various formal language patterns

### Custom Languages (S1-S4)
- **S1-S4**: Specialized sequence patterns

### Data Format
CSV files with format: `value1,value2,...,valueN,label`
- Sequences are normalized using z-score per sequence
- Labels are binary (0.0 or 1.0)
- Padding applied to `max_len` with normalized value 0.0

## 🏗️ Model Architectures

### LSTM Classifier
```python
# Architecture
Embedding Layer → LSTM → Average Pooling → Classifier → BCEWithLogitsLoss
```

**Features:**
- LSTM with configurable hidden size
- Average pooling over valid (non-padded) tokens
- Dropout for regularization
- AdamW optimizer with weight decay

### Transformer Encoder
```python
# Architecture  
Input Projection → Positional Embedding → Transformer Encoder → Average Pooling → Classifier → BCEWithLogitsLoss
```

**Features:**
- Multi-head self-attention
- Positional embeddings for sequence order
- Padding mask handling for variable-length sequences
- Layer normalization and residual connections

## 🎯 Training Strategy

Both models use identical training procedures:

1. **Step-based Training**: Maximum 30,000 steps (not epochs)
2. **Early Stopping**: Stop after achieving target accuracy 3 consecutive times
3. **Evaluation**: Every 100 steps with comprehensive metrics
4. **Optimization**: AdamW with learning rate 0.001 and weight decay 0.01
5. **Loss Function**: BCEWithLogitsLoss for stable training

### Training Flow
```
Data Loading → 80/20 Train/Val Split → Step-based Training → Early Stopping → Test Evaluation
```

## 📈 Performance Metrics

Both models provide detailed evaluation:

```
Confusion Matrix:
  True Positives  (TP): 1000
  False Positives (FP): 0  
  True Negatives  (TN): 1000
  False Negatives (FN): 0

Performance Metrics:
  Accuracy:  1.000000 (100.0000%)
  Precision: 1.000000 (100.0000%)
  Recall:    1.000000 (100.0000%)
  F1-Score:  1.000000 (100.0000%)
```

## 🔬 Examples

### Basic Training
```bash
# Train LSTM on L1 with defaults
python3 lstm/lstm.py --language L1

# Train Transformer on S2 with custom parameters  
python3 transformer/transformer_encoder.py --language S2 --d_model 256 --num_layers 2
```

### Model Loading and Inference
```bash
# Continue training from checkpoint
python3 lstm/lstm.py --language L1 --load_model lstm_best_model_L1_1.0000.pt

# Inference only (no training)
python3 transformer/transformer_encoder.py --language L1 --load_model transformer_best_model_L1_1.0000.pt --inference_only
```

### Practical Perfect Accuracy
```bash
# Use 99.95% as target instead of 100%
python3 lstm/lstm.py --language L2 --practical_perfect
```

## 🛠️ Implementation Details

### Data Preprocessing
- **Z-score Normalization**: Each sequence normalized independently `(x - mean) / std`
- **Padding**: Sequences padded/truncated to `max_len` 
- **Padding Value**: 0.0 (normalized mean)

### Model Outputs
- **Training**: Models output logits (raw values)
- **Loss**: BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
- **Evaluation**: Sigmoid applied to logits, threshold at 0.5

### Reproducibility
- Fixed random seeds (SEED=42)
- Deterministic operations where possible
- Identical train/validation splits

## 📋 Results Format

Models save results to:
- **Model Files**: `{model}_best_model_{language}_{accuracy}.pt`
- **Log Files**: `{model}_evaluation_{language}.log`

Log files contain:
- Model configuration
- Training parameters  
- Comprehensive test metrics
- Validation set comparison
- Timestamp and device information

## 🤝 Contributing

The codebase is designed for:
- Easy model comparison
- Extensible architecture  
- Clear parameter management
- Comprehensive evaluation

## 📝 Notes

- Both models achieve near-perfect performance on included datasets
- Training typically converges in 100-1000 steps for simple languages
- More complex languages may require longer training
- Models handle variable-length sequences through padding masks