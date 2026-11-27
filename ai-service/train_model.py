from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.features import MODEL_FEATURE_ORDER, RAW_FEATURE_ORDER, model_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_inventory_risk.csv"
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORT_DIR = ROOT / "reports"

FEATURE_ORDER = list(RAW_FEATURE_ORDER)
LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
CLASS_INDICES = [LABEL_MAP[label] for label in CLASS_ORDER]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    required = FEATURE_ORDER + ["risk"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains null values")

    features = df[FEATURE_ORDER].astype(float)
    labels = df["risk"].map(LABEL_MAP)
    if labels.isna().any():
        raise ValueError("Dataset contains invalid risk labels")
    return features, labels.astype(int)


def rule_agreement(df: pd.DataFrame) -> float:
    sales_rate = 0.7 * df["averageMonthlySales"] + 0.3 * df["recentSales"]
    expected = sales_rate * (df["daysUntilExpiry"] / 30)
    rule = np.select(
        [df["currentStock"] > 2 * expected, df["currentStock"] > expected],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    return float((rule == df["risk"]).mean())


def clean_classification_report(report: dict) -> dict:
    cleaned = {}
    for label, values in report.items():
        if isinstance(values, dict):
            cleaned[label] = {metric: float(value) for metric, value in values.items()}
        else:
            cleaned[label] = float(values)
    return cleaned


def evaluate_model(
    name: str,
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_INDICES,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "name": name,
        "model": model,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_INDICES).tolist(),
        "classification_report": clean_classification_report(report),
    }


def save_confusion_matrix(result: dict, path: Path) -> None:
    matrix = np.array(result["confusion_matrix"])
    figure, axes = plt.subplots(figsize=(5, 4))
    image = axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axes.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(f"Confusion Matrix: {result['name']}")
    for row in range(len(CLASS_ORDER)):
        for column in range(len(CLASS_ORDER)):
            axes.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_feature_importance(result: dict, path: Path) -> None:
    importances = np.asarray(result["model"].feature_importances_, dtype=float)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.barh(MODEL_FEATURE_ORDER, importances)
    axes.set_xlabel("Importance")
    axes.set_title(f"Feature Importance: {result['name']}")
    axes.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_metrics(
    df: pd.DataFrame,
    results: list[dict],
    best: dict,
    agreement: float,
) -> None:
    payload = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "rows": int(len(df)),
            "class_counts": {label: int((df["risk"] == label).sum()) for label in CLASS_ORDER},
            "rule_agreement": agreement,
        },
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "models": {
            result["name"]: {
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
                "confusion_matrix": result["confusion_matrix"],
                "classification_report": result["classification_report"],
            }
            for result in results
        },
        "chosen_model": {
            "name": best["name"],
            "accuracy": best["accuracy"],
            "f1_macro": best["f1_macro"],
            "selection_rule": "highest test F1 macro, then highest accuracy",
            "model_feature_order": list(MODEL_FEATURE_ORDER),
        },
        "feature_importance": {
            feature: float(importance)
            for feature, importance in zip(
                MODEL_FEATURE_ORDER, best["model"].feature_importances_
            )
        },
    }
    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_metadata(best: dict) -> None:
    metadata = {
        "model_type": best["name"],
        "raw_feature_order": FEATURE_ORDER,
        "model_feature_order": list(MODEL_FEATURE_ORDER),
        "label_map": LABEL_MAP,
        "reverse_label_map": REVERSE_LABEL_MAP,
        "random_state": RANDOM_STATE,
        "accuracy": best["accuracy"],
        "f1_macro": best["f1_macro"],
    }
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_report(df: pd.DataFrame, results: list[dict], best: dict, agreement: float) -> None:
    lines = [
        "# Model Comparison Report",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(df)}",
        f"- Rule agreement with noisy labels: {agreement:.2%}",
        f"- Train/test split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, stratified, seed {RANDOM_STATE}",
        "",
        "## Comparison",
        "",
        "| Model | Accuracy | F1 macro | F1 weighted |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['accuracy']:.4f} | "
            f"{result['f1_macro']:.4f} | {result['f1_weighted']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Chosen model: {best['name']}",
            "",
            "Selected by highest test F1 macro, then highest accuracy.",
            "",
            "## Confusion matrix (chosen model)",
            "",
            "| Actual / Predicted | LOW | MEDIUM | HIGH |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in zip(CLASS_ORDER, best["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Feature importance (chosen model)",
            "",
            "The model receives the four API fields plus derived demand features: "
            + ", ".join(MODEL_FEATURE_ORDER)
            + ".",
            "",
            "| Feature | Importance |",
            "|---|---:|",
        ]
    )
    for feature, importance in zip(MODEL_FEATURE_ORDER, best["model"].feature_importances_):
        lines.append(f"| {feature} | {float(importance):.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The dataset is synthetic and labels contain controlled noise near rule boundaries, "
            "so very high accuracy would suggest leakage or an overly memorised rule.",
            "",
            f"Charts: `confusion_matrix.png`, `feature_importance.png`. "
            f"Raw metrics: `metrics.json`.",
            "",
        ]
    )
    with (REPORT_DIR / "model_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    agreement = rule_agreement(df)
    features, labels = load_data(DATA_PATH)
    model_input = model_features(features.to_numpy())
    x_train, x_test, y_train, y_test = train_test_split(
        model_input,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    candidates = [
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in candidates
    ]
    best = max(results, key=lambda result: (result["f1_macro"], result["accuracy"]))

    joblib.dump(best["model"], MODEL_PATH)
    write_metadata(best)
    write_metrics(df, results, best, agreement)
    write_report(df, results, best, agreement)
    save_confusion_matrix(best, REPORT_DIR / "confusion_matrix.png")
    save_feature_importance(best, REPORT_DIR / "feature_importance.png")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Chosen model: {best['name']}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"F1 macro: {best['f1_macro']:.4f}")
    print(f"Rule agreement: {agreement:.4f}")
    for result in results:
        print(
            f"{result['name']}: accuracy={result['accuracy']:.4f} "
            f"f1_macro={result['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
