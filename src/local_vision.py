"""Local SIGAP inference using the project's own trained PyTorch checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.model import ARCHITECTURE, build_model

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "sigap_mobilenetv3_best.pt"
MODEL_ID = "SIGAP custom-trained MobileNetV3-Small"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LocalModel:
    model: torch.nn.Module
    device: torch.device
    transform: Any
    class_names: list[str]
    metadata: dict[str, Any]
    model_path: Path


def interpret_label(label: str) -> dict[str, str]:
    """Convert a model label to cautious user-facing visual screening guidance."""
    normalized = str(label).strip().lower().replace("_", " ")
    if "rotten" in normalized or "spoiled" in normalized or "decay" in normalized:
        return {
            "kind": "possible_spoilage",
            "title": "Possible signs of spoilage",
            "message": "The trained model classified this image closer to its Rotten class. Inspect the item and follow food-safety guidance; the model result is not a definitive safety verdict.",
        }
    if "fresh" in normalized:
        return {
            "kind": "fresh_prediction",
            "title": "Model predicts: Fresh",
            "message": "The trained model grouped this image with its Fresh class. A photo cannot establish that the food is safe to eat.",
        }
    return {
        "kind": "cannot_assess",
        "title": "Cannot interpret result",
        "message": "The trained model returned an unfamiliar label. Retake the photo or inspect the item using appropriate food-safety guidance.",
    }


def build_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    """Preprocessing must match the validation transforms used during training."""
    resize_size = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_local_model(model_path: str | Path | None = None, device: str | None = None) -> LocalModel:
    """Load the model checkpoint created by scripts/train_model.py.

    Inference does not contact any hosted inference API. The checkpoint must be
    present locally; pretrained ImageNet weights are only needed when training.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"SIGAP has no trained checkpoint at {path}. Run scripts/train_model.py first."
        )

    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        checkpoint = torch.load(path, map_location=chosen_device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions supported by local installs
        checkpoint = torch.load(path, map_location=chosen_device)

    architecture = checkpoint.get("architecture", ARCHITECTURE)
    if architecture != ARCHITECTURE:
        raise ValueError(f"Unsupported model architecture in checkpoint: {architecture}")

    class_names = checkpoint.get("class_names", ["Fresh", "Rotten"])
    if not isinstance(class_names, list) or len(class_names) != 2:
        raise ValueError("Checkpoint must contain exactly two class names: Fresh and Rotten.")

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(chosen_device)
    model.eval()

    metadata = {
        key: checkpoint.get(key)
        for key in ("created_at", "training_config", "validation_metrics", "test_metrics", "dataset_source")
        if checkpoint.get(key) is not None
    }
    return LocalModel(
        model=model,
        device=chosen_device,
        transform=build_eval_transform(image_size),
        class_names=[str(item) for item in class_names],
        metadata=metadata,
        model_path=path,
    )


def predict_photo(image_bytes: bytes, learner: LocalModel) -> dict[str, Any]:
    """Classify an uploaded image using a local SIGAP checkpoint."""
    with Image.open(BytesIO(image_bytes)) as image_file:
        image = image_file.convert("RGB")

    tensor = learner.transform(image).unsqueeze(0).to(learner.device)
    with torch.inference_mode():
        logits = learner.model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()

    probabilities = [float(value) for value in probabilities_tensor.tolist()]
    pred_index = int(probabilities_tensor.argmax().item())
    label = learner.class_names[pred_index] if pred_index < len(learner.class_names) else f"Class {pred_index}"
    ranked = sorted(
        [
            {"label": class_name, "score": probabilities[index]}
            for index, class_name in enumerate(learner.class_names)
            if index < len(probabilities)
        ],
        key=lambda row: row["score"],
        reverse=True,
    )
    return {
        "label": label,
        "score": probabilities[pred_index] if pred_index < len(probabilities) else 0.0,
        "assessment": interpret_label(label),
        "alternatives": ranked,
        "model_name": MODEL_ID,
        "model_path": str(learner.model_path),
        "metadata": learner.metadata,
    }
