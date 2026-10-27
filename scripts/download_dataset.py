"""Download and export the raw Project-AgML fruit freshness dataset.

The raw Hugging Face dataset is several GB. For lower storage usage, users can
instead prepare data/raw/Fresh and data/raw/Rotten manually and skip this script.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ID = "Project-AgML/fresh_rotten_fruit_classification"
DATASET_CONFIG = "raw"


def parse_label(dataset, value) -> str:
    try:
        label = dataset.features["label"].int2str(value)
    except (AttributeError, KeyError, TypeError, ValueError):
        label = str(value)
    normalized = str(label).strip().lower()
    if normalized in {"fresh", "0"}:
        return "Fresh"
    if normalized in {"rotten", "spoiled", "1"}:
        return "Rotten"
    raise ValueError(f"Unknown label in dataset: {value!r} ({label!r})")


def export_dataset(output_dir: Path, max_per_class: int | None = None, overwrite: bool = False) -> dict[str, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies with: pip install -r requirements-train.txt") from exc

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.rglob("*.jpg")):
        if not overwrite:
            raise FileExistsError(
                f"Images already exist in {output_dir}. Use --overwrite to replace this export, "
                "or select another --output-dir."
            )
        shutil.rmtree(output_dir)

    (output_dir / "Fresh").mkdir(parents=True, exist_ok=True)
    (output_dir / "Rotten").mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} (config={DATASET_CONFIG}) from Hugging Face...")
    print("First download can use several GB of disk space; only raw, non-augmented images are selected.")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    counts = {"Fresh": 0, "Rotten": 0}
    seen_hashes: dict[str, str] = {}
    duplicates_skipped = 0
    bad_rows_skipped = 0
    conflicting_label_duplicates = 0

    for row_index, row in enumerate(dataset):
        try:
            label = parse_label(dataset, row["label"])
            if max_per_class is not None and counts[label] >= max_per_class:
                if all(value >= max_per_class for value in counts.values()):
                    break
                continue
            image = row["image"].convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen_hashes:
                duplicates_skipped += 1
                if seen_hashes[digest] != label:
                    conflicting_label_duplicates += 1
                continue
            seen_hashes[digest] = label
            fruit_type = str(row.get("fruit_type", "fruit")).strip().lower().replace(" ", "_")
            safe_type = "".join(ch for ch in fruit_type if ch.isalnum() or ch in "_-" )[:24] or "fruit"
            counts[label] += 1
            out = output_dir / label / f"{safe_type}_{counts[label]:05d}_{digest[:8]}.jpg"
            image.save(out, format="JPEG", quality=94, optimize=True)
        except (AttributeError, OSError, ValueError, KeyError) as exc:
            bad_rows_skipped += 1
            print(f"Skipping row {row_index}: {type(exc).__name__}")

        if (row_index + 1) % 250 == 0:
            print(f"Processed {row_index + 1}/{len(dataset)} rows · Fresh={counts['Fresh']} · Rotten={counts['Rotten']}")

    if min(counts.values()) < 5:
        raise RuntimeError(f"Too few images exported: {counts}. Check dataset access or label mapping.")

    summary = {
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "license": "CC BY 4.0",
        "counts": counts,
        "exact_pixel_duplicates_skipped": duplicates_skipped,
        "conflicting_label_duplicates": conflicting_label_duplicates,
        "unreadable_or_invalid_rows_skipped": bad_rows_skipped,
        "source_rows": len(dataset),
    }
    (output_dir / "dataset_info.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nDataset export complete:")
    print(f"  Fresh:  {counts['Fresh']}")
    print(f"  Rotten: {counts['Rotten']}")
    print(f"  Exact pixel duplicates skipped: {duplicates_skipped}")
    print(f"  Duplicate image hashes with conflicting labels: {conflicting_label_duplicates}")
    print(f"  Unreadable / invalid rows skipped: {bad_rows_skipped}")
    print(f"  Output: {output_dir}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional exported-image cap per class")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing JPG export")
    args = parser.parse_args()
    if args.max_per_class is not None and args.max_per_class < 5:
        parser.error("--max-per-class must be at least 10 so the exported dataset has 20 images for train/validation/test splits")
    export_dataset(args.output_dir, args.max_per_class, args.overwrite)


if __name__ == "__main__":
    main()
