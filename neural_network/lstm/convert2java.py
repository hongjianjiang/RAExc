import torch
from lstm import LSTMClassifier  # Make sure this points to your model definition
import numpy as np
def convert_to_jit(model_path: str, output_jit_path: str, max_len: int = 51):
    """
    Load a saved PyTorch model and export it as a TorchScript JIT model.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt).
        output_jit_path: Path to save the JIT-compiled model.
        max_len: Input sequence length used during training.
    """
    device = torch.device('cpu')  # JIT tracing should happen on CPU for portability

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    # Rebuild the model
    model = LSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_path}")
    
    # Create dummy input for tracing
    dummy_input = torch.zeros(1, max_len, dtype=torch.float32)  # batch_size=1

    # Trace the model
    traced = torch.jit.trace(model, dummy_input)
    
    # Save the TorchScript model
    traced.save(output_jit_path)
    print(f"JIT model saved to {output_jit_path}")
    
def predict(model_path, seq):
    """
    Predict using the JIT model.
    
    Args:
        input_data: List of input values to predict.
    """
    device = torch.device('cpu')  # JIT tracing should happen on CPU for portability

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    # Rebuild the model
    model = LSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = np.mean(seq)
    std = np.std(seq)
    if std == 0:
        norm_seq = [0.0 for _ in seq]
    else:
        norm_seq = [(x - mean) / std for x in seq]
    
    padded_seq = norm_seq + [-999.0] * (51 - len(norm_seq))
    x = torch.tensor([padded_seq], dtype=torch.float).to("cpu")
    with torch.no_grad():
        prob = model(x).item()  # Model already outputs probabilities with sigmoid
    print(f"Probability: {prob}")
    return prob >= 0.5 
    
if __name__ == "__main__":
    seqs = [[1,2]]
    # Probability: 7.966403359205287e-08
    # -> 9.213552E-17
    model = "lstm_best_model_S8_0.9999.pt"
    jit = "S8.pt"
    # seqs = [[137,137,137,137,137,137,137,14,14,-148,137,137,14,14,14,137,14,14,14]]
    convert_to_jit(model, jit, max_len=51)

    for seq in seqs:
        # seq = [-59.54756876201619,-45.52060694634636,73.03100173718502,-96.8895984177254]
        print(predict(model, seq))