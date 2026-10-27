from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_model import make_splits, compute_metrics, make_transforms  # noqa: E402


def test_stratified_splits_are_disjoint_and_cover_all_rows():
    labels = np.array([0] * 50 + [1] * 50)
    train, val, test = make_splits(labels, seed=42)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == len(labels)
    assert set(np.unique(labels[train])) == {0, 1}
    assert set(np.unique(labels[val])) == {0, 1}
    assert set(np.unique(labels[test])) == {0, 1}


def test_splits_require_both_classes_with_enough_images():
    with pytest.raises(ValueError, match="at least 5"):
        make_splits([0] * 16 + [1] * 4)


def test_metrics_include_confusion_matrix_and_macro_f1():
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["Fresh", "Rotten"])
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_transforms_are_defined_for_train_and_eval():
    train_transform, eval_transform = make_transforms(224)
    assert callable(train_transform)
    assert callable(eval_transform)
