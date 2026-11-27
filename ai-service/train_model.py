from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.features import MODEL_FEATURE_ORDER, RAW_FEATURE_ORDER, model_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_inventory_risk.csv"
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORT_DIR = ROOT / "reports"

LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
CLASS_INDICES = [LABEL_MAP[label] for label in CLASS_ORDER]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    required = list(RAW_FEATURE_ORDER) + ["risk"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains null values")

    features = df[list(RAW_FEATURE_ORDER)].astype(float)
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


def slugify(name: str) -> str:
    return "_".join(name.lower().replace("-", " ").split())


def format_params(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def build_candidates() -> list[tuple[str, object, dict]]:
    return [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [8, 12, None],
                "min_samples_leaf": [2, 5, 10],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            {
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 2, 3],
            },
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            {
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 2, 3],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.1],
                "max_depth": [2, 3],
            },
        ),
        (
            "Hist Gradient Boosting",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "max_iter": [100, 200],
                "max_leaf_nodes": [15, 31],
            },
        ),
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("estimator", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ]
            ),
            {"estimator__C": [0.1, 1, 10]},
        ),
    ]


def tune_model(
    name: str,
    estimator: object,
    parameter_grid: dict,
    x_train: np.ndarray,
    y_train: pd.Series,
) -> dict:
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return {
        "name": name,
        "slug": slugify(name),
        "model": search.best_estimator_,
        "parameter_grid": parameter_grid,
        "best_params": search.best_params_,
        "cv_f1_macro": float(search.best_score_),
    }


def evaluate_model(result: dict, x_test: np.ndarray, y_test: pd.Series) -> None:
    predictions = result["model"].predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_INDICES,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    result["accuracy"] = float(accuracy_score(y_test, predictions))
    result["f1_macro"] = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    result["f1_weighted"] = float(
        f1_score(y_test, predictions, average="weighted", zero_division=0)
    )
    result["confusion_matrix"] = confusion_matrix(
        y_test, predictions, labels=CLASS_INDICES
    ).tolist()
    result["classification_report"] = clean_classification_report(report)


def feature_importance(
    result: dict, x_test: np.ndarray, y_test: pd.Series
) -> tuple[np.ndarray, str]:
    model = result["model"]
    native_model = model.named_steps["estimator"] if isinstance(model, Pipeline) else model
    if hasattr(native_model, "feature_importances_"):
        return np.asarray(native_model.feature_importances_, dtype=float), "native"

    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return np.asarray(permutation.importances_mean, dtype=float), "permutation"


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
    importances = np.asarray(result["feature_importance"], dtype=float)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.barh(MODEL_FEATURE_ORDER, importances)
    axes.set_xlabel(result["importance_source"] + " importance")
    axes.set_title(f"Feature Importance: {result['name']}")
    axes.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_artifacts(best: dict, results: list[dict], x_test: np.ndarray, y_test: pd.Series) -> None:
    for result in results:
        save_confusion_matrix(result, REPORT_DIR / f"confusion_matrix_{result['slug']}.png")
        save_feature_importance(result, REPORT_DIR / f"feature_importance_{result['slug']}.png")

    best_result = next(result for result in results if result["name"] == best["name"])
    save_confusion_matrix(best_result, REPORT_DIR / "confusion_matrix.png")
    save_feature_importance(best_result, REPORT_DIR / "feature_importance.png")


def write_metrics(df: pd.DataFrame, results: list[dict], best: dict, agreement: float) -> None:
    payload = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT)),
            "rows": int(len(df)),
            "class_counts": {label: int((df["risk"] == label).sum()) for label in CLASS_ORDER},
            "rule_agreement": agreement,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "cv_scoring": "f1_macro",
        },
        "models": {
            result["name"]: {
                "parameter_grid": {key: str(value) for key, value in result["parameter_grid"].items()},
                "best_params": {key: str(value) for key, value in result["best_params"].items()},
                "cv_f1_macro": result["cv_f1_macro"],
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
                "confusion_matrix": result["confusion_matrix"],
                "classification_report": result["classification_report"],
                "importance_source": result["importance_source"],
                "feature_importance": {
                    feature: float(importance)
                    for feature, importance in zip(
                        MODEL_FEATURE_ORDER, result["feature_importance"]
                    )
                },
            }
            for result in results
        },
        "chosen_model": {
            "name": best["name"],
            "accuracy": best["accuracy"],
            "f1_macro": best["f1_macro"],
            "selection_rule": "highest test F1 macro, then highest accuracy, then highest CV F1 macro",
            "best_params": {key: str(value) for key, value in best["best_params"].items()},
        },
    }
    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_metadata(best: dict) -> None:
    metadata = {
        "model_type": best["name"],
        "raw_feature_order": list(RAW_FEATURE_ORDER),
        "model_feature_order": list(MODEL_FEATURE_ORDER),
        "label_map": LABEL_MAP,
        "reverse_label_map": REVERSE_LABEL_MAP,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "best_params": {key: str(value) for key, value in best["best_params"].items()},
        "accuracy": best["accuracy"],
        "f1_macro": best["f1_macro"],
        "cv_f1_macro": best["cv_f1_macro"],
    }
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_comparison_csv(results: list[dict]) -> None:
    rows = [
        {
            "model": result["name"],
            "best_params": format_params(result["best_params"]),
            "cv_f1_macro": result["cv_f1_macro"],
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["f1_weighted"],
            "importance_source": result["importance_source"],
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_comparison.csv", index=False)


def write_hyperparameter_report(results: list[dict]) -> None:
    lines = [
        "# Hyperparameter Comparison",
        "",
        f"Grid search used {CV_FOLDS}-fold stratified cross-validation on the training split, "
        "scored by macro F1. Each selected configuration was then evaluated once on the held-out "
        "test split.",
        "",
        "| Model | Grid | Best parameters | CV F1 macro | Test F1 macro | Test accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        grid = ", ".join(
            f"{key}={value}" for key, value in result["parameter_grid"].items()
        )
        lines.append(
            f"| {result['name']} | {grid} | {format_params(result['best_params'])} | "
            f"{result['cv_f1_macro']:.4f} | {result['f1_macro']:.4f} | "
            f"{result['accuracy']:.4f} |"
        )
    with (REPORT_DIR / "hyperparameter_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_confusion_matrices_report(results: list[dict]) -> None:
    lines = ["# Confusion Matrices", ""]
    for result in results:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                "| Actual / Predicted | LOW | MEDIUM | HIGH |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, row in zip(CLASS_ORDER, result["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
        lines.append("")
    with (REPORT_DIR / "confusion_matrices.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_report(df: pd.DataFrame, results: list[dict], best: dict, agreement: float) -> None:
    lines = [
        "# Model Comparison Report",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(df)}",
        f"- Rule agreement with noisy labels: {agreement:.2%}",
        f"- Train/test split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, stratified, seed {RANDOM_STATE}",
        f"- Hyperparameter search: {CV_FOLDS}-fold stratified CV, macro F1 scoring",
        "",
        "## Models tested",
        "",
        ", ".join(result["name"] for result in results),
        "",
        "## Comparison",
        "",
        "| Model | CV F1 macro | Accuracy | F1 macro | F1 weighted |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['cv_f1_macro']:.4f} | "
            f"{result['accuracy']:.4f} | {result['f1_macro']:.4f} | "
            f"{result['f1_weighted']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Chosen model: {best['name']}",
            "",
            f"- Best parameters: {format_params(best['best_params'])}",
            f"- Test accuracy: {best['accuracy']:.4f}",
            f"- Test macro F1: {best['f1_macro']:.4f}",
            "- Selection rule: highest test F1 macro, then highest accuracy, then CV F1 macro.",
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
            f"Source: {best['importance_source']}.",
            "The model receives the four API fields plus derived demand features: "
            + ", ".join(MODEL_FEATURE_ORDER)
            + ".",
            "",
            "| Feature | Importance |",
            "|---|---:|",
        ]
    )
    for feature, importance in zip(MODEL_FEATURE_ORDER, best["feature_importance"]):
        lines.append(f"| {feature} | {float(importance):.4f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The data is synthetic and labels include controlled boundary noise, so accuracy "
            "should stay below the 93.27% rule agreement. A near-perfect score would suggest "
            "leakage or a memorised rule.",
            "",
            "## Artifacts",
            "",
            "- `metrics.json`: full metrics and parameters for every model",
            "- `model_comparison.csv`: spreadsheet-friendly comparison",
            "- `hyperparameter_comparison.md`: grids, best parameters, CV and test scores",
            "- `confusion_matrices.md`: confusion matrix for every model",
            "- `confusion_matrix_<model>.png`: confusion matrix chart for every model",
            "- `feature_importance_<model>.png`: importance chart for every model",
            "- `confusion_matrix.png` and `feature_importance.png`: chosen model charts",
            "",
        ]
    )
    with (REPORT_DIR / "model_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_reports_index(results: list[dict]) -> None:
    lines = [
        "# Reports Index",
        "",
        "Generated by `ai-service/train_model.py`.",
        "",
        "- `model_comparison.md`: readable summary and chosen model",
        "- `metrics.json`: full metrics for every model",
        "- `model_comparison.csv`: spreadsheet-friendly comparison",
        "- `hyperparameter_comparison.md`: parameter grids and CV results",
        "- `confusion_matrices.md`: confusion matrix tables for every model",
        "- `confusion_matrix.png`: chosen model confusion matrix chart",
        "- `feature_importance.png`: chosen model feature importance chart",
        "",
        "## Per-model charts",
        "",
    ]
    for result in results:
        lines.append(
            f"- {result['name']}: `confusion_matrix_{result['slug']}.png`, "
            f"`feature_importance_{result['slug']}.png`"
        )
    with (REPORT_DIR / "REPORTS_INDEX.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


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

    results = [
        tune_model(name, estimator, parameter_grid, x_train, y_train)
        for name, estimator, parameter_grid in build_candidates()
    ]

    for result in results:
        evaluate_model(result, x_test, y_test)
        importances, source = feature_importance(result, x_test, y_test)
        result["feature_importance"] = importances
        result["importance_source"] = source

    best = max(
        results,
        key=lambda result: (
            result["f1_macro"],
            result["accuracy"],
            result["cv_f1_macro"],
        ),
    )

    joblib.dump(best["model"], MODEL_PATH)
    write_metadata(best)
    write_metrics(df, results, best, agreement)
    write_comparison_csv(results)
    write_hyperparameter_report(results)
    write_confusion_matrices_report(results)
    write_report(df, results, best, agreement)
    write_reports_index(results)
    save_artifacts(best, results, x_test, y_test)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Chosen model: {best['name']}")
    print(f"Best parameters: {format_params(best['best_params'])}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"F1 macro: {best['f1_macro']:.4f}")
    print(f"Rule agreement: {agreement:.4f}")
    for result in results:
        print(
            f"{result['name']}: accuracy={result['accuracy']:.4f} "
            f"f1_macro={result['f1_macro']:.4f} cv_f1={result['cv_f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
