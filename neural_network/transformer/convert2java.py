import torch
from transformer_encoder import TransformerClassifier  # Make sure this points to your model definition
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
    model = TransformerClassifier(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_len']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_path}")
    
    # Create dummy input for tracing - use realistic values instead of zeros
    dummy_input = torch.randn(1, max_len, dtype=torch.float32)  # Random values instead of zeros
    
    # Use torch.jit.script instead of trace for better compatibility
    try:
        # First try scripting (more robust)
        scripted = torch.jit.script(model)
        scripted.save(output_jit_path)
        print(f"JIT model saved using scripting to {output_jit_path}")
    except Exception as script_error:
        print(f"Scripting failed: {script_error}")
        print("Falling back to tracing...")
        
        try:
            # Fall back to tracing with multiple examples
            traced = torch.jit.trace(model, dummy_input, strict=False)
            traced.save(output_jit_path)
            print(f"JIT model saved using tracing to {output_jit_path}")
        except Exception as trace_error:
            print(f"Tracing also failed: {trace_error}")
            
            # Try a different approach - save a wrapper
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    return self.model(x)
            
            wrapper = ModelWrapper(model)
            traced_wrapper = torch.jit.trace(wrapper, dummy_input, strict=False)
            traced_wrapper.save(output_jit_path)
            print(f"JIT model saved using wrapper to {output_jit_path}")
    
def predict_jit(jit_model_path, seq):
    """
    Predict using the JIT model directly.
    
    Args:
        jit_model_path: Path to the JIT model file.
        seq: List of input values to predict.
    """
    device = torch.device('cpu')
    
    # Load JIT model
    try:
        jit_model = torch.jit.load(jit_model_path, map_location=device)
        jit_model.eval()
        print(f"JIT model loaded from {jit_model_path}")
    except Exception as e:
        print(f"Failed to load JIT model: {e}")
        return None

    # Preprocessing (same as regular model)
    mean = np.mean(seq)
    std = np.std(seq)
    if std == 0:
        norm_seq = [0.0 for _ in seq]
    else:
        norm_seq = [(x - mean) / std for x in seq]
    
    padded_seq = norm_seq + [-999.0] * (51 - len(norm_seq))
    x = torch.tensor([padded_seq], dtype=torch.float).to(device)
    
    with torch.no_grad():
        logits = jit_model(x)  # Model outputs logits
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        pred = (probs >= 0.5).float()  # Convert to binary predictions
    
    print(f"JIT Logits: {logits.item()}")
    print(f"JIT Probability: {probs.item()}")
    print(f"JIT Prediction: {pred.item()}")
    return pred.item()

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
    model = TransformerClassifier(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_len']
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
        logits = model(x)  # Model outputs logits
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        pred = (probs >= 0.5).float()  # Convert to binary predictions
    # print(f"Logits: {logits.item()}")
    # print(f"Probability: {probs.item()}")
    print(f"Prediction: {pred.item()}")
    return pred.item()

if __name__ == "__main__":
    seqs = [ 
          [1,2]#0
 ]


    model_path = "transformer_best_model_S6_0.9950.pt"
    jit_path = "S6.pt"

    # Convert to JIT
    convert_to_jit(model_path, jit_path, max_len=51)
    
    # Test both models for comparison
    for i, seq in enumerate(seqs):
        # print(f"\n--- Testing sequence {i+1}: {seq} ---")
        # print("Original model:")
        orig_pred = predict(model_path, seq)
        
        # print("JIT model:")
        # jit_pred = predict_jit(jit_path, seq)
        
        # if jit_pred is not None:
        #     print(f"Results match: {orig_pred == jit_pred}")
        # print("-" * 50)