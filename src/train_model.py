"""Train and evaluate the Food Waste Prediction models.

The supplied dataset is small and its label is strongly separated by the
remaining-stock percentage. Metrics are therefore descriptive of this dataset,
not evidence of real-world performance.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_penjualan.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "food_waste_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
IMAGE_DIR = BASE_DIR / "images"
TREE_IMAGE_PATH = IMAGE_DIR / "decision_tree.png"

FEATURES = ["stok", "terjual", "hari_ke", "cuaca", "hari_besar", "sisa_persen"]
TARGET = "label"
REQUIRED_COLUMNS = FEATURES + [TARGET]
RANDOM_STATE = 42


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the training dataset; fail with actionable errors."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Add data_penjualan.csv to the data/ folder."
        )

    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[REQUIRED_COLUMNS].isna().any().any():
        bad_columns = data[REQUIRED_COLUMNS].columns[data[REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {', '.join(bad_columns)}")
    if not np.isfinite(data[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite values. Replace them with valid numbers.")
    if data["stok"].le(0).any():
        raise ValueError("All stock values must be greater than 0.")
    if data["terjual"].lt(0).any() or data["terjual"].gt(data["stok"]).any():
        raise ValueError("Sold values must be between 0 and available stock.")
    if not data["hari_ke"].between(1, 30).all():
        raise ValueError("hari_ke must be between 1 and 30.")
    for column in ["cuaca", "hari_besar", TARGET]:
        if not set(data[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data[TARGET].nunique() != 2:
        raise ValueError("Training needs examples of both labels: 0 (Safe) and 1 (Potential Waste).")

    expected_remaining = ((data["stok"] - data["terjual"]) / data["stok"] * 100).round(2)
    if not np.allclose(data["sisa_persen"], expected_remaining, atol=0.02):
        raise ValueError(
            "sisa_persen does not match (stok - terjual) / stok × 100. "
            "Check the dataset before training."
        )
    return data


def _candidate_models() -> dict[str, Any]:
    """Candidates deliberately remain interpretable and suitable for a small dataset."""
    return {
        "Decision Tree (depth 2)": DecisionTreeClassifier(
            criterion="entropy", max_depth=2, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree (depth 4)": DecisionTreeClassifier(
            criterion="entropy", max_depth=4, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    }


def _score_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> dict[str, Any]:
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
            "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        },
        n_jobs=1,
        error_score="raise",
    )
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"], ddof=0)),
            "folds": [float(x) for x in scores[f"test_{metric}"]],
        }
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    }


def train_and_save_model() -> dict[str, Any]:
    """Compare models with CV on the training split, then evaluate once on holdout."""
    data = load_dataset()
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidate_scores: dict[str, dict[str, Any]] = {}
    candidates = _candidate_models()
    for name, estimator in candidates.items():
        try:
            candidate_scores[name] = _score_model(estimator, X_train, y_train, cv)
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Skipping {name} because CV failed: {exc}")

    if not candidate_scores:
        raise RuntimeError("No candidate model could be evaluated. Check dataset size and labels.")

    # Best mean macro-F1; for near-ties (within 0.005), prefer the simpler, more
    # interpretable model. The baseline is never preferred over a real classifier on a tie.
    best_score = max(v["f1_macro"]["mean"] for v in candidate_scores.values())
    near_best = [
        name for name, score in candidate_scores.items()
        if score["f1_macro"]["mean"] >= best_score - 0.005
        and name != "Majority-class baseline"
    ]
    if near_best:
        preferred_order = ["Decision Tree (depth 2)", "Decision Tree (depth 4)", "Random Forest"]
        selected_name = next((name for name in preferred_order if name in near_best), near_best[0])
    else:
        selected_name = max(candidate_scores, key=lambda name: candidate_scores[name]["f1_macro"]["mean"])

    model = candidates[selected_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Safe", "Potential Waste"],
        output_dict=True, zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "features": FEATURES,
        "classes": {0: "Safe", 1: "Potential Waste"},
        "model_name": selected_name,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    selected_cv = candidate_scores[selected_name]
    label_counts = y.value_counts().sort_index()
    metadata: dict[str, Any] = {
        "model": selected_name,
        "selection_metric": "mean macro-F1 across 5-fold stratified CV on training split",
        "criterion": getattr(model, "criterion", None),
        "max_depth": getattr(model, "max_depth", None),
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "class_counts": {"safe": int(label_counts.get(0, 0)), "potential_waste": int(label_counts.get(1, 0))},
        "cross_validation": {key: {"mean": value["mean"], "std": value["std"]} for key, value in selected_cv.items()},
        "candidate_comparison": {
            name: {key: {"mean": value["mean"], "std": value["std"]} for key, value in scores.items()}
            for name, scores in candidate_scores.items()
        },
        "holdout_metrics": metrics,
        "confusion_matrix_labels": ["Safe", "Potential Waste"],
        "confusion_matrix": matrix,
        "classification_report": report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_quality": {
            "duplicate_rows": int(data.duplicated().sum()),
            "remaining_percentage_is_derived_from_stock_and_sales": True,
            "label_separation_warning": (
                "Labels are fully separated by remaining-stock percentage in the supplied dataset. "
                "This may reflect a rule used to create the labels rather than generalizable real-world behaviour."
            ),
        },
        "scope_note": (
            "Metrics describe this small supplied dataset only. They are not a guarantee of performance "
            "on real inventory data. Model confidence is not a calibrated probability of food waste."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Decision Tree models can be visualized directly. If a forest is chosen, use its first tree.
    tree_model = model if isinstance(model, DecisionTreeClassifier) else getattr(model, "estimators_", [None])[0]
    if tree_model is not None and hasattr(tree_model, "tree_"):
        plt.figure(figsize=(15, 8))
        plot_tree(
            tree_model, feature_names=FEATURES, class_names=["Safe", "Potential Waste"],
            filled=True, rounded=True, impurity=False, proportion=True,
            max_depth=5, fontsize=8,
        )
        plt.title(f"{selected_name} — fitted on training split", pad=16)
        plt.tight_layout()
        plt.savefig(TREE_IMAGE_PATH, dpi=180, bbox_inches="tight")
        plt.close()

    return metadata


if __name__ == "__main__":
    result = train_and_save_model()
    print("Model training and evaluation completed.")
    print(f"Selected model: {result['model']}")
    print(f"Holdout accuracy: {result['holdout_metrics']['accuracy']:.2%}")
    print(f"Holdout macro-F1: {result['holdout_metrics']['f1_macro']:.2%}")
    print(f"5-fold CV macro-F1: {result['cross_validation']['f1_macro']['mean']:.2%} ± {result['cross_validation']['f1_macro']['std']:.2%}")
    print(f"Saved model: {MODEL_PATH}")
