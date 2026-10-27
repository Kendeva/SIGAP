"""Fine-tune MobileNetV3-Small on Fresh/Rotten images for local SIGAP use."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import ARCHITECTURE, build_model  # noqa: E402

DATASET_SOURCE = "Project-AgML/fresh_rotten_fruit_classification (raw config, CC BY 4.0)"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_CLASSES = ["Fresh", "Rotten"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    resize_size = int(round(image_size * 256 / 224))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_splits(labels: list[int] | np.ndarray, seed: int = 42):
    """Create 70/15/15 stratified splits from original images, not augmented copies."""
    labels = np.asarray(labels, dtype=int)
    indices = np.arange(len(labels))
    if len(indices) < 20:
        raise ValueError("At least 20 images are required to create train/validation/test splits.")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2 or counts.min() < 5:
        raise ValueError("Need at least 5 original images in each of the Fresh and Rotten classes.")
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=labels
    )
    val_relative = 0.15 / 0.85
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def validate_dataset(data_dir: Path) -> ImageFolder:
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\nRun: python scripts/download_dataset.py"
        )
    dataset = ImageFolder(str(data_dir))
    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            "Dataset must contain exactly these class folders (case-sensitive): "
            f"{EXPECTED_CLASSES}. Found: {dataset.classes}"
        )
    targets = [sample[1] for sample in dataset.samples]
    if len(dataset) < 20:
        raise ValueError(f"Only {len(dataset)} images found; at least 20 are needed for a basic split.")
    for class_index, class_name in enumerate(dataset.classes):
        count = sum(label == class_index for label in targets)
        if count < 5:
            raise ValueError(f"Class {class_name} has only {count} images; provide at least 5 per class.")
    # Verify files early so corrupt images do not cause a surprise halfway through an epoch.
    corrupt = []
    for filepath, _ in dataset.samples:
        try:
            with Image.open(filepath) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            corrupt.append(filepath)
            if len(corrupt) >= 10:
                break
    if corrupt:
        raise ValueError("Unreadable image files found, for example: " + ", ".join(corrupt[:3]))
    return dataset


def compute_metrics(labels: list[int], predictions: list[int], class_names: list[str]) -> dict:
    report = classification_report(
        labels, predictions, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(class_names)))).tolist(),
        "classification_report": report,
    }


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool) -> tuple[float, list[int], list[int]]:
    model.train(train)
    total_loss = 0.0
    total_items = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
            preds_all.extend(int(value) for value in preds.detach().cpu().tolist())
    return total_loss / max(total_items, 1), labels_all, preds_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 by default for Windows compatibility")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet pretrained weights (not recommended for real training)")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64 or args.patience < 1:
        parser.error("epochs/batch-size/patience must be positive and image-size must be >= 64")
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training selected. This is supported but may take longer than training on a GPU.")
    print(f"Reading images from: {data_dir}")
    base_dataset = validate_dataset(data_dir)
    class_names = list(base_dataset.classes)
    targets = np.asarray([target for _, target in base_dataset.samples], dtype=int)
    train_idx, val_idx, test_idx = make_splits(targets, seed=args.seed)
    print(f"Dataset images: {len(base_dataset)}")
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print("Split ratios are approximately 70/15/15; test data is held out until final evaluation.")

    train_transform, eval_transform = make_transforms(args.image_size)
    train_data = ImageFolder(str(data_dir), transform=train_transform)
    eval_data = ImageFolder(str(data_dir), transform=eval_transform)
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Subset(eval_data, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Subset(eval_data, test_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    print("Building MobileNetV3-Small...")
    if args.no_pretrained:
        print("WARNING: Training without pretrained ImageNet weights. Results may be much weaker.")
    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained).to(device)

    # Class-weighted loss helps when users provide an imbalanced custom dataset.
    train_labels = targets[train_idx]
    counts = np.bincount(train_labels, minlength=len(class_names)).astype(float)
    weights = len(train_labels) / (len(class_names) * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    checkpoint_path = output_dir / "sigap_mobilenetv3_best.pt"
    history_path = output_dir / "training_history.csv"
    history: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_y, val_pred = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        val_metrics = compute_metrics(val_y, val_pred, class_names)
        scheduler.step(val_metrics["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": round(time.time() - epoch_started, 2),
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_metrics['accuracy']:.3f} | "
            f"val macro-F1 {val_metrics['macro_f1']:.3f} | {record['epoch_seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "architecture": ARCHITECTURE,
                "num_classes": len(class_names),
                "class_names": class_names,
                "image_size": args.image_size,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_source": DATASET_SOURCE if (data_dir / "dataset_info.json").exists() else "User-provided local image folders",
                "training_config": {
                    "epochs_requested": args.epochs,
                    "best_epoch": best_epoch,
                    "batch_size": args.batch_size,
                    "image_size": args.image_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "pretrained_imagenet_weights": not args.no_pretrained,
                    "train_size": int(len(train_idx)),
                    "validation_size": int(len(val_idx)),
                    "test_size": int(len(test_idx)),
                },
                "validation_metrics": val_metrics,
                "test_metrics": None,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
            break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Load the best validation checkpoint and evaluate the held-out test split exactly once.
    try:
        best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_loss, test_y, test_pred = run_epoch(model, test_loader, loss_fn, optimizer=None, device=device, train=False)
    test_metrics = compute_metrics(test_y, test_pred, class_names)
    best["test_metrics"] = test_metrics
    best["test_loss"] = test_loss
    torch.save(best, checkpoint_path)

    summary = {
        "model_path": str(checkpoint_path),
        "architecture": ARCHITECTURE,
        "class_names": class_names,
        "dataset_source": best["dataset_source"],
        "created_at": best["created_at"],
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_images": int(len(train_idx)),
        "validation_images": int(len(val_idx)),
        "test_images": int(len(test_idx)),
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
        "test_loss": test_loss,
        "training_seconds": round(time.time() - started, 2),
        "caveat": "Scores describe this held-out split only. They do not prove reliable food-safety detection on real-world foods outside the dataset distribution.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHeld-out test results (not used for model selection):")
    print(json.dumps({key: test_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "confusion_matrix")}, indent=2))
    print(f"\nSaved local model: {checkpoint_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved training history: {history_path}")
    print("Remember: a photo-based prediction is a screening aid, not a food-safety verdict.")


if __name__ == "__main__":
    main()
