import io
import torch

def count_parameters_ts(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total


def load_torchscript_model_from_upload(uploaded_file, device='cpu'):
    """
    Load TorchScript model from Streamlit UploadedFile
    """
    model_bytes = uploaded_file.getvalue()
    buffer = io.BytesIO(model_bytes)
    model = torch.jit.load(buffer, map_location=device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def load_torchscript_model_smart(model_source, device='cpu'):
    """
    Smart model loader that accepts EITHER:
    - String path (for local testing)
    - Streamlit UploadedFile (for web app)
    
    Args:
        model_source: str path OR Streamlit UploadedFile object
        device: Device to load model on
    
    Returns:
        tuple: (model, parameter_count)
    """
    # Check if it's a string path
    if isinstance(model_source, str):
        model = torch.jit.load(model_source, map_location=device)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        return model, param_count
    
    # Otherwise, assume it's a Streamlit UploadedFile
    else:
        return load_torchscript_model_from_upload(model_source, device)
