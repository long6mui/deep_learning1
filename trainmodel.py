from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "predictions"
TRAIN_PATH = DATASET_DIR / "train.pkl"
TEST_PATH = DATASET_DIR / "test.pkl"
TRAIN_PRED_PATH = OUTPUT_DIR / "train_oof_predictions.csv"
TEST_PRED_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
MODEL_PATH = OUTPUT_DIR / "catboost_model.cbm"


BASE_MODEL_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass
class PreparedDataset:
    ids: list[str]
    frame: pd.DataFrame
    target: np.ndarray | None
    categorical_columns: list[str]


def load_pickle(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def extract_first_observation(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    if not value:
        return None

    normalized: list[tuple[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            item_value = item.get("value")
            charttime = str(item.get("charttime") or "")
            if item_value is not None:
                normalized.append((charttime, item_value))
        elif item is not None:
            normalized.append(("", item))

    if not normalized:
        return None

    normalized.sort(key=lambda pair: pair[0])
    return normalized[0][1]


def build_feature_list(
    train_data: dict[str, dict[str, Any]], test_data: dict[str, dict[str, Any]]
) -> list[str]:
    train_features = set(next(iter(train_data.values())).keys()) - {"target"}
    test_features = set(next(iter(test_data.values())).keys())

    # Keep only the shared feature space so train and test are consistent.
    return sorted(train_features & test_features)


def detect_categorical_columns(frame: pd.DataFrame) -> list[str]:
    categorical_columns: list[str] = []
    for column in frame.columns:
        non_null_values = frame[column].dropna()
        if non_null_values.empty:
            continue
        if any(isinstance(value, str) for value in non_null_values.iloc[:50]):
            categorical_columns.append(column)
    return categorical_columns


def prepare_dataset(
    raw_data: dict[str, dict[str, Any]],
    features: list[str],
) -> PreparedDataset:
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    targets: list[int] = []

    for patient_id, record in raw_data.items():
        ids.append(str(patient_id))
        row = {feature: extract_first_observation(record.get(feature)) for feature in features}
        rows.append(row)
        if "target" in record:
            targets.append(int(record["target"]))

    frame = pd.DataFrame(rows, index=ids)
    categorical_columns = detect_categorical_columns(frame)

    for column in categorical_columns:
        frame[column] = frame[column].fillna("MISSING").astype(str)

    return PreparedDataset(
        ids=ids,
        frame=frame,
        target=np.asarray(targets, dtype=np.int64) if targets else None,
        categorical_columns=categorical_columns,
    )


def create_model(**overrides: Any) -> CatBoostClassifier:
    params = dict(BASE_MODEL_PARAMS)
    params.update(overrides)
    return CatBoostClassifier(**params)


def choose_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(target, probabilities)
    scores = tpr - fpr
    best_index = int(np.argmax(scores))
    threshold = float(thresholds[best_index])
    if not np.isfinite(threshold):
        threshold = 0.5
    return threshold


def train_and_predict(
    train_dataset: PreparedDataset,
    test_dataset: PreparedDataset,
) -> dict[str, Any]:
    x_train = train_dataset.frame
    y_train = train_dataset.target
    x_test = test_dataset.frame
    categorical_columns = train_dataset.categorical_columns

    if y_train is None:
        raise ValueError("Training dataset does not contain target labels.")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probabilities = np.zeros(len(x_train), dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_index, (fit_idx, valid_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        x_fit = x_train.iloc[fit_idx]
        y_fit = y_train[fit_idx]
        x_valid = x_train.iloc[valid_idx]
        y_valid = y_train[valid_idx]

        model = create_model()
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_valid, y_valid),
            cat_features=categorical_columns,
            use_best_model=True,
            early_stopping_rounds=100,
        )

        valid_probabilities = model.predict_proba(x_valid)[:, 1]
        oof_probabilities[valid_idx] = valid_probabilities

        best_iteration = model.get_best_iteration()
        best_iteration = int(best_iteration if best_iteration and best_iteration > 0 else BASE_MODEL_PARAMS["iterations"])
        best_iterations.append(best_iteration)

        fold_metrics.append(
            {
                "fold": fold_index,
                "auc": round(float(roc_auc_score(y_valid, valid_probabilities)), 6),
                "best_iteration": best_iteration,
            }
        )

    oof_auc = float(roc_auc_score(y_train, oof_probabilities))
    threshold = choose_threshold(y_train, oof_probabilities)
    final_iterations = int(round(float(np.median(best_iterations)))) if best_iterations else BASE_MODEL_PARAMS["iterations"]

    final_model = create_model(iterations=final_iterations)
    final_model.fit(x_train, y_train, cat_features=categorical_columns)

    train_prediction_labels = (oof_probabilities >= threshold).astype(int)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    test_prediction_labels = (test_probabilities >= threshold).astype(int)

    final_model.save_model(str(MODEL_PATH))

    train_predictions = pd.DataFrame(
        {
            "id": train_dataset.ids,
            "probability": oof_probabilities,
            "prediction_label": train_prediction_labels,
        }
    )
    test_predictions = pd.DataFrame(
        {
            "id": test_dataset.ids,
            "probability": test_probabilities,
            "prediction_label": test_prediction_labels,
        }
    )

    train_predictions.to_csv(TRAIN_PRED_PATH, index=False)
    test_predictions.to_csv(TEST_PRED_PATH, index=False)

    metrics = {
        "oof_auc": round(oof_auc, 6),
        "threshold": round(threshold, 6),
        "final_iterations": final_iterations,
        "feature_count": len(x_train.columns),
        "categorical_columns": categorical_columns,
        "fold_metrics": fold_metrics,
        "train_output": str(TRAIN_PRED_PATH),
        "test_output": str(TEST_PRED_PATH),
        "model_output": str(MODEL_PATH),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_train = load_pickle(TRAIN_PATH)
    raw_test = load_pickle(TEST_PATH)
    features = build_feature_list(raw_train, raw_test)

    train_dataset = prepare_dataset(raw_train, features)
    test_dataset = prepare_dataset(raw_test, features)
    metrics = train_and_predict(train_dataset, test_dataset)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
