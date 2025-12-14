import json
import torchvision.transforms as T


ALLOWED = {
    # geometry / sizing
    "Resize",
    "CenterCrop",
    "Pad",

    # tensor conversion + normalization (required core)
    "ToTensor",
    "Normalize",

    # quality-of-life (still safe)
    "Grayscale",
    "ConvertImageDtype",
}
def default_transform():
    return T.ToTensor()


def load_transform_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

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
