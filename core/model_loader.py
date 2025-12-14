import torch

def count_parameters_ts(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total

def load_torchscript_model(model_path, device="cpu", max_params=15_000_000):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load TorchScript model: {e}")

    param_count = count_parameters_ts(model)
    if param_count > max_params:
        raise ValueError(
            f"Model has {param_count/1e6:.2f}M parameters. Max allowed is 15M."
        )

    return model, param_count
