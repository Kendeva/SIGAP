"""Validated single-row and batch prediction helpers for the Streamlit app."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.joblib"
INPUT_COLUMNS = ["product_name", "stok", "terjual", "hari_ke", "cuaca", "hari_besar"]
FEATURES_FALLBACK = ["stok", "terjual", "hari_ke", "cuaca", "hari_besar", "sisa_persen"]


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load the trained model once per process; clear cache after model retraining."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run 'python src/train_model.py' first.")
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("The saved model file is incomplete. Run 'python src/train_model.py' again.")
    return bundle


def clear_model_cache() -> None:
    """Useful for tests or when a model is retrained while the app process is running."""
    load_model_bundle.cache_clear()


def validate_inventory(stock: Any, sold: Any, day_index: Any, weather_code: Any, special_day: Any) -> None:
    try:
        stock_f, sold_f = float(stock), float(sold)
        day_f, weather_f = float(day_index), float(weather_code)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inventory values must be numeric.") from exc
    if not np.isfinite([stock_f, sold_f, day_f, weather_f]).all():
        raise ValueError("Inventory values must be finite numbers.")
    if not stock_f.is_integer() or not sold_f.is_integer():
        raise ValueError("Stock and sold quantities must be whole numbers.")
    if stock_f <= 0:
        raise ValueError("Stock must be greater than 0.")
    if sold_f < 0 or sold_f > stock_f:
        raise ValueError("Items sold must be between 0 and available stock.")
    if not day_f.is_integer() or not 1 <= day_f <= 30:
        raise ValueError("Day index must be a whole number between 1 and 30.")
    if not weather_f.is_integer() or weather_f not in (0, 1):
        raise ValueError("Weather code must be 0 or 1.")
    if isinstance(special_day, (bool, np.bool_)):
        pass
    elif isinstance(special_day, (int, float, np.integer, np.floating)):
        if float(special_day) not in (0.0, 1.0):
            raise ValueError("Special-day value must be boolean or 0/1.")
    elif isinstance(special_day, str):
        if special_day.strip().lower() not in {"0", "1", "true", "false", "yes", "no"}:
            raise ValueError("Special-day value must be boolean or 0/1.")
    else:
        raise ValueError("Special-day value must be boolean or 0/1.")


def build_recommendation(prediction: int, remaining_percentage: float, remaining: int) -> str:
    """Rule-based suggestion, intentionally distinct from the ML prediction."""
    if remaining <= 0:
        return "No stock remains. Review whether the current reorder point is appropriate."
    if prediction == 1:
        if remaining_percentage >= 60:
            return "High remaining stock: review the next replenishment and prioritize selling existing inventory."
        if remaining_percentage >= 35:
            return "Potential waste flagged: check recent sales, avoid unnecessary replenishment, and consider a targeted promotion."
        return "Potential waste flagged despite a lower remaining share: review product context and sales before taking action."
    if remaining_percentage <= 15:
        return "Low remaining stock: monitor availability and consider replenishment if demand is expected to continue."
    return "No elevated waste risk was flagged by this model. Continue monitoring sales and stock before the next replenishment."


def _model_input(stock: int, sold: int, day_index: int, weather_code: int, special_day: bool, features: list[str]) -> pd.DataFrame:
    remaining = stock - sold
    remaining_percentage = round((remaining / stock) * 100, 2)
    row = {
        "stok": stock,
        "terjual": sold,
        "hari_ke": day_index,
        "cuaca": weather_code,
        "hari_besar": int(bool(special_day)),
        "sisa_persen": remaining_percentage,
    }
    missing = [name for name in features if name not in row]
    if missing:
        raise ValueError(f"The saved model expects unsupported features: {', '.join(missing)}")
    return pd.DataFrame([{name: row[name] for name in features}], columns=features)


def _predict_validated(stock: int, sold: int, day_index: int, weather_code: int, special_day: bool) -> dict[str, Any]:
    validate_inventory(stock, sold, day_index, weather_code, special_day)
    stock, sold, day_index, weather_code = int(stock), int(sold), int(day_index), int(weather_code)
    special_day = _normalize_special_day(special_day)
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle.get("features", FEATURES_FALLBACK)
    frame = _model_input(stock, sold, day_index, weather_code, bool(special_day), features)
    prediction = int(model.predict(frame)[0])
    remaining = stock - sold
    remaining_percentage = (remaining / stock) * 100

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        if prediction in classes:
            confidence = float(probabilities[classes.index(prediction)])

    return {
        "prediction": prediction,
        "status": "Potential Waste" if prediction == 1 else "Safe",
        "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK",
        "stock": stock,
        "sold": sold,
        "remaining": remaining,
        "remaining_percentage": remaining_percentage,
        "confidence": confidence,
        "recommendation": build_recommendation(prediction, remaining_percentage, remaining),
        "model_name": bundle.get("model_name", type(model).__name__),
    }


def predict_food_waste(stock: int, sold: int, day_index: int, weather_code: int, special_day: bool) -> dict[str, Any]:
    """Predict a single inventory record after validating business constraints."""
    return _predict_validated(stock, sold, day_index, weather_code, special_day)


def _normalize_special_day(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError("missing special-day value")
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in (0.0, 1.0):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError("hari_besar must be 0/1, true/false, or yes/no")


def predict_batch(data: pd.DataFrame) -> pd.DataFrame:
    """Predict valid rows and preserve invalid rows with a row-level explanation.

    Required input columns: stok, terjual, hari_ke, cuaca, hari_besar. The
    optional product_name column identifies rows. sisa_persen is always derived.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Batch input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    required = ["stok", "terjual", "hari_ke", "cuaca", "hari_besar"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if len(data) > 10000:
        raise ValueError("This prototype supports up to 10,000 rows per upload.")

    rows: list[dict[str, Any]] = []
    for idx, source_row in data.reset_index(drop=True).iterrows():
        product_name = str(source_row.get("product_name", f"Product {idx + 1}")).strip()
        if not product_name or product_name.lower() == "nan":
            product_name = f"Product {idx + 1}"
        output: dict[str, Any] = {"product_name": product_name, "source_row": idx + 2}
        try:
            stock = float(source_row["stok"])
            sold = float(source_row["terjual"])
            day = float(source_row["hari_ke"])
            weather = float(source_row["cuaca"])
            special = _normalize_special_day(source_row["hari_besar"])
            validate_inventory(stock, sold, day, weather, special)
            if not all(number.is_integer() for number in [stock, sold, day, weather]):
                raise ValueError("stok, terjual, hari_ke, and cuaca must be whole numbers")
            result = _predict_validated(int(stock), int(sold), int(day), int(weather), special)
            output.update({
                "status": result["status"],
                "risk_level": result["risk_label"],
                "stok": result["stock"],
                "terjual": result["sold"],
                "sisa_stok": result["remaining"],
                "sisa_persen": round(result["remaining_percentage"], 2),
                "model_confidence": result["confidence"],
                "recommendation": result["recommendation"],
                "validation_status": "Valid",
                "validation_error": "",
            })
        except (ValueError, TypeError, OverflowError) as exc:
            output.update({
                "status": "Not evaluated",
                "risk_level": "—",
                "stok": source_row.get("stok"),
                "terjual": source_row.get("terjual"),
                "sisa_stok": None,
                "sisa_persen": None,
                "model_confidence": None,
                "recommendation": "Fix the row data and upload again.",
                "validation_status": "Invalid",
                "validation_error": str(exc),
            })
        rows.append(output)
    return pd.DataFrame(rows)
