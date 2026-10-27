"""Regression tests for dataset validation and prediction input handling."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictor import predict_batch, predict_food_waste, validate_inventory  # noqa: E402
from src.train_model import load_dataset  # noqa: E402


def test_dataset_loads_and_derived_percentage_is_consistent():
    data = load_dataset()
    assert len(data) == 80
    expected = ((data["stok"] - data["terjual"]) / data["stok"] * 100).round(2)
    assert (data["sisa_persen"].sub(expected).abs() <= 0.02).all()


@pytest.mark.parametrize(
    "values",
    [
        (0, 0, 1, 0, False),
        (10, -1, 1, 0, False),
        (10, 11, 1, 0, False),
        (10, 1, 0, 0, False),
        (10, 1, 31, 0, False),
        (10, 1, 1, 2, False),
        (10.5, 1, 1, 0, False),
        (10, 1, 1, 0, 2),
    ],
)
def test_invalid_inventory_is_rejected(values):
    with pytest.raises(ValueError):
        validate_inventory(*values)


def test_valid_inventory_is_accepted():
    validate_inventory(100, 40, 15, 0, False)


def test_batch_keeps_invalid_row_and_explains_error(monkeypatch):
    # Mock the prediction function path indirectly by supplying one invalid row;
    # it should be handled at row level and not abort processing the batch.
    data = pd.DataFrame([
        {"product_name": "Bad stock", "stok": 0, "terjual": 1, "hari_ke": 5, "cuaca": 0, "hari_besar": 0},
    ])
    result = predict_batch(data)
    assert len(result) == 1
    assert result.loc[0, "validation_status"] == "Invalid"
    assert "Stock" in result.loc[0, "validation_error"]


def test_batch_requires_core_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        predict_batch(pd.DataFrame({"stok": [10]}))


def test_batch_marks_fractional_integer_fields_invalid():
    data = pd.DataFrame([
        {"stok": 10.5, "terjual": 1, "hari_ke": 5, "cuaca": 0, "hari_besar": 0},
    ])
    result = predict_batch(data)
    assert result.loc[0, "validation_status"] == "Invalid"
    assert "whole numbers" in result.loc[0, "validation_error"]


def test_single_prediction_returns_expected_inventory_math():
    result = predict_food_waste(100, 40, 15, 0, False)
    assert result["stock"] == 100
    assert result["remaining"] == 60
    assert result["remaining_percentage"] == pytest.approx(60.0)
    assert result["status"] in {"Safe", "Potential Waste"}
    assert result["recommendation"]


def test_batch_can_process_valid_row_and_numeric_binary_float():
    data = pd.DataFrame([
        {"product_name": "Bread", "stok": 100, "terjual": 40, "hari_ke": 15, "cuaca": 0.0, "hari_besar": 0.0},
    ])
    result = predict_batch(data)
    assert result.loc[0, "validation_status"] == "Valid"
    assert result.loc[0, "sisa_stok"] == 60
    assert result.loc[0, "sisa_persen"] == pytest.approx(60.0)


def test_string_false_special_day_is_not_treated_as_true():
    false_result = predict_food_waste(100, 40, 15, 0, "false")
    true_result = predict_food_waste(100, 40, 15, 0, "true")
    # Both are valid booleans, and they must be handled as different input values.
    # The result can coincidentally match due to the fitted tree; verify the normalization
    # helper directly to avoid asserting a dataset-dependent prediction difference.
    from src.predictor import _normalize_special_day
    assert _normalize_special_day("false") is False
    assert _normalize_special_day("true") is True
    assert false_result["stock"] == true_result["stock"] == 100


def test_batch_accepts_english_csv_column_names():
    data = pd.DataFrame([
        {"product_name": "Apple", "stock": 100, "sold": 40, "day_index": 15, "weather_code": 0, "special_day": 0},
    ])
    result = predict_batch(data)
    assert len(result) == 1
    assert result.loc[0, "validation_status"] == "Valid"
    assert result.loc[0, "sisa_stok"] == 60
    assert result.loc[0, "sisa_persen"] == pytest.approx(60.0)
