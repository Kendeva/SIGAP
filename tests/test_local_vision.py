from __future__ import annotations
from io import BytesIO
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_vision import DEFAULT_MODEL_PATH, LocalModel, interpret_label, predict_photo, load_local_model  # noqa: E402


def test_fresh_label_does_not_claim_food_is_safe():
    result = interpret_label("Fresh")
    assert result["kind"] == "fresh_prediction"
    assert "cannot establish" in result["message"]


def test_rotten_label_flags_possible_spoilage():
    result = interpret_label("Rotten")
    assert result["kind"] == "possible_spoilage"
    assert "Possible signs" in result["title"]


def test_unknown_label_falls_back_cautiously():
    assert interpret_label("unexpected-class")["kind"] == "cannot_assess"


def test_load_model_explains_missing_custom_checkpoint(tmp_path):
    missing = tmp_path / "not-trained-yet.pt"
    try:
        load_local_model(missing, device="cpu")
    except FileNotFoundError as exc:
        assert "scripts/train_model.py" in str(exc)
    else:
        raise AssertionError("Missing checkpoint should not silently fall back to a different model")


class DummyTransform:
    def __call__(self, image):
        return torch.zeros((3, 224, 224), dtype=torch.float32)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.0, 2.0]], dtype=torch.float32).repeat(x.shape[0], 1)


def test_predict_photo_returns_top_label_and_scores():
    image_buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(120, 90, 50)).save(image_buffer, format="JPEG")
    bundle = LocalModel(
        model=DummyModel(),
        device=torch.device("cpu"),
        transform=DummyTransform(),
        class_names=["Fresh", "Rotten"],
        metadata={"best_epoch": 2},
        model_path=DEFAULT_MODEL_PATH,
    )
    result = predict_photo(image_buffer.getvalue(), bundle)
    assert result["label"] == "Rotten"
    assert result["assessment"]["kind"] == "possible_spoilage"
    assert len(result["alternatives"]) == 2
    assert result["score"] > 0.5
