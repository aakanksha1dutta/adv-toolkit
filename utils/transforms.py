import json
import torchvision.transforms as T

ALLOWED = {
    # geometry / sizing
    "Resize",
    "CenterCrop",
    "Pad",

    "ToTensor",
    "Normalize",

    "Grayscale",
    "ConvertImageDtype",
}

def default_transform():
    return T.ToTensor()


def load_transform_from_json(json_path):
    """Load transform from JSON file path"""
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return _create_transform_from_config(cfg)


def load_transform_from_upload(uploaded_file):
    """Load transform from Streamlit UploadedFile"""
    json_str = uploaded_file.getvalue().decode('utf-8')
    cfg = json.loads(json_str)
    return _create_transform_from_config(cfg)


def load_transform_smart(transform_source):
    """
    Smart transform loader that accepts EITHER:
    - String path (for local testing)
    - Streamlit UploadedFile (for web app)
    
    Args:
        transform_source: str path OR Streamlit UploadedFile object
    
    Returns:
        Transform object
    """
    # Check if it's a string path
    if isinstance(transform_source, str):
        return load_transform_from_json(transform_source)
    # Otherwise, assume it's a Streamlit UploadedFile
    else:
        return load_transform_from_upload(transform_source)


def _create_transform_from_config(cfg):
    """
    Internal helper to create transform from config dictionary
    
    Args:
        cfg: Dictionary with transform configuration
    
    Returns:
        Composed transform
    """
    steps = cfg["transforms"] if isinstance(cfg, dict) and "transforms" in cfg else cfg
    
    if not isinstance(steps, list):
        raise ValueError("Transform JSON must be a list or an object with key 'transforms'.")
    
    t = []
    FORBIDDEN = {"ToTensor"}
    
    for step in steps:
        name = step["name"]
        params = step.get("params", {}) or {}
        
        if name in FORBIDDEN:
            raise ValueError(
                f"`{name}` is not allowed. Images are already tensors. "
                "User transforms must be tensor-compatible."
            )
        
        try:
            transform_cls = getattr(T, name)
        except AttributeError:
            raise ValueError(f"Unknown torchvision transform: {name}")
        
        t.append(transform_cls(**params))
    
    return T.Compose(t)
