from __future__ import annotations

import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

TEAM_NAME = "PowerNextAI"
DATASET_PATH = Path("Dataset.xlsx")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
SUBMISSION_DIR = Path(f"{TEAM_NAME}-submission")
SEED = 42
ID_COLUMN = "Test_ID"
REGRESSION_TARGET = "Reference_Parameter"
CLASSIFICATION_TARGET = "Validity_Label"
N_CLASSIFICATION_REPEATS = 5  # average OOF probabilities over several shuffles for a stabler threshold


def make_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    )


def identify_sheets(path: Path):
    workbook = pd.ExcelFile(path)
    frames = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in workbook.sheet_names}
    train_name = next(
        (name for name, frame in frames.items() if REGRESSION_TARGET in frame.columns and CLASSIFICATION_TARGET in frame.columns),
        None,
    )
    test_name = next(
        (name for name, frame in frames.items() if ID_COLUMN in frame.columns and REGRESSION_TARGET not in frame.columns and CLASSIFICATION_TARGET not in frame.columns and name != train_name),
        None,
    )
    if train_name is None or test_name is None:
        raise ValueError("Could not identify labeled training and unlabeled test sheets")
    return frames, train_name, test_name


def validate_inputs(train: pd.DataFrame, test: pd.DataFrame):
    required_train = {ID_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Training data is missing columns: {required_train - set(train.columns)}")
    if REGRESSION_TARGET in test.columns or CLASSIFICATION_TARGET in test.columns:
        raise ValueError("Test data contains target columns")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate Test_ID values detected")
    if train.empty or test.empty:
        raise ValueError("Training and test data must both be non-empty")


def add_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Base feature engineering, now with extra signals aimed at invalid-record detection:
    per-sensor pairwise gaps (a physically inconsistent sensor sticks out in a gap, not just
    the mean/std), a per-row missingness count, and a coefficient-of-variation across sensors."""
    result = frame[feature_columns].copy()
    sensors = [column for column in feature_columns if column.startswith("Sensor_")]
    if len(sensors) >= 2:
        result["Sensor_mean"] = result[sensors].mean(axis=1)
        result["Sensor_std"] = result[sensors].std(axis=1)
        result["Sensor_range"] = result[sensors].max(axis=1) - result[sensors].min(axis=1)
        result["Sensor_cv"] = result["Sensor_std"] / result["Sensor_mean"].replace(0, np.nan).abs()
        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                result[f"{sensors[i]}_minus_{sensors[j]}"] = result[sensors[i]] - result[sensors[j]]
    if {"Load_Current_A", "Applied_Voltage_kV"}.issubset(result.columns):
        result["Electrical_power_proxy"] = result["Load_Current_A"] * result["Applied_Voltage_kV"]
    if {"Ambient_Temperature_C", "Test_Duration_min"}.issubset(result.columns):
        result["Temperature_duration_proxy"] = result["Ambient_Temperature_C"] * result["Test_Duration_min"]
    result["Missing_count"] = frame[feature_columns].isna().sum(axis=1)
    return result


def add_anomaly_features(x_train: pd.DataFrame, x_test: pd.DataFrame, y_reg: pd.Series):
    """Two model-based anomaly signals, computed leak-free and available for both train and
    test (neither needs the true Reference_Parameter, so nothing here uses labels at inference
    time):

    1. Regressor disagreement: ExtraTrees and RandomForest regressors are fit independently and
       their predictions compared. Rows where two different models substantially disagree on the
       expected Reference_Parameter tend to be exactly the rows with an inconsistent/noisy sensor
       reading -- a strong, previously-unused invalid-record signal. Train values are produced
       out-of-fold via cross_val_predict so the classifier never sees in-fold information.
    2. Isolation Forest anomaly score: an unsupervised outlier score over the full engineered
       feature space, fit on the (imputed) training rows only and scored on both splits.
    """
    imputer = SimpleImputer(strategy="median")
    x_train_imp = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    reg_a = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.9, random_state=SEED, n_jobs=1)
    reg_b = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.9, random_state=SEED, n_jobs=1)
    oof_a = cross_val_predict(reg_a, x_train_imp, y_reg, cv=cv)
    oof_b = cross_val_predict(reg_b, x_train_imp, y_reg, cv=cv)
    reg_a.fit(x_train_imp, y_reg)
    reg_b.fit(x_train_imp, y_reg)
    test_a = reg_a.predict(x_test_imp)
    test_b = reg_b.predict(x_test_imp)

    train_disagreement = np.abs(oof_a - oof_b)
    test_disagreement = np.abs(test_a - test_b)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=1)
    iso.fit(x_train_imp)
    train_anomaly = -iso.decision_function(x_train_imp)  # flip sign so higher = more anomalous
    test_anomaly = -iso.decision_function(x_test_imp)

    x_train_out = x_train.copy()
    x_test_out = x_test.copy()
    x_train_out["Reg_disagreement"] = train_disagreement
    x_test_out["Reg_disagreement"] = test_disagreement
    x_train_out["Isolation_anomaly_score"] = train_anomaly
    x_test_out["Isolation_anomaly_score"] = test_anomaly
    return x_train_out, x_test_out


def best_f1_threshold(y_true, probabilities) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    f1 = f1[:-1]  # precision_recall_curve returns one extra point with no matching threshold
    if len(f1) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate_models(x_train, y_reg, y_class):
    regression_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "ExtraTreesRegressor": make_pipeline(
            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9, random_state=SEED, n_jobs=1)
        ),
        "RandomForestRegressor": make_pipeline(
            RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9, random_state=SEED, n_jobs=1)
        ),
    }
    comparison = []
    for name, model in candidates.items():
        prediction = cross_val_predict(model, x_train, y_reg, cv=regression_cv, n_jobs=None)
        comparison.append(
            {
                "task": "regression",
                "model": name,
                "mae": mean_absolute_error(y_reg, prediction),
                "rmse": mean_squared_error(y_reg, prediction) ** 0.5,
                "r2": r2_score(y_reg, prediction),
            }
        )

    # --- Classification: try a couple of balanced tree ensembles, average OOF probabilities
    # across several shuffled splits to stabilize the threshold choice, then pick the
    # (model, threshold) combination that maximizes invalid-class F1 out-of-fold. ---
    classifier_candidates = {
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.9, class_weight="balanced", random_state=SEED, n_jobs=1
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features=0.9, class_weight="balanced_subsample", random_state=SEED, n_jobs=1
        ),
    }

    best_choice = None
    for name, base_model in classifier_candidates.items():
        prob_accumulator = np.zeros(len(y_class))
        for repeat in range(N_CLASSIFICATION_REPEATS):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
            model = make_pipeline(base_model)
            probs = cross_val_predict(model, x_train, y_class, cv=cv, method="predict_proba")[:, 1]
            prob_accumulator += probs
        avg_probs = prob_accumulator / N_CLASSIFICATION_REPEATS
        threshold, oof_f1 = best_f1_threshold(y_class, avg_probs)
        prediction = (avg_probs >= threshold).astype(int)
        candidate_result = {
            "task": "classification",
            "model": name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_class, prediction),
            "invalid_precision": precision_score(y_class, prediction, zero_division=0),
            "invalid_recall": recall_score(y_class, prediction, zero_division=0),
            "invalid_f1": f1_score(y_class, prediction, zero_division=0),
            "confusion_matrix_valid_invalid": confusion_matrix(y_class, prediction, labels=[0, 1]).tolist(),
        }
        comparison.append(candidate_result)
        if best_choice is None or candidate_result["invalid_f1"] > best_choice["invalid_f1"]:
            best_choice = candidate_result
            best_classifier_pipeline = make_pipeline(classifier_candidates[name])
            best_threshold = threshold

    return pd.DataFrame(comparison), candidates["ExtraTreesRegressor"], best_classifier_pipeline, best_threshold, best_choice["model"]


def write_methodology(path: Path, metrics: dict, train_shape: tuple[int, int], test_shape: tuple[int, int], features: list[str]):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.4, leading=10.2, spaceAfter=3))
    styles["Title"].fontSize = 16
    styles["Title"].leading = 18
    story = [Paragraph("PowerNext-AI Screening Submission: Methodology", styles["Title"]), Spacer(1, 3 * mm)]
    body = [
        f"<b>Problem and data.</b> The workbook contains {train_shape[0]} labeled training rows and {test_shape[0]} unlabeled test rows. The training targets are Reference_Parameter (regression) and Validity_Label (binary classification). Test_ID is retained only for output and is excluded from modeling.",
        "<b>Preprocessing and features.</b> Missing numeric values are median-imputed inside each scikit-learn pipeline, with missingness indicators. Derived features include sensor mean, standard deviation, range, coefficient of variation, all pairwise sensor gaps, a per-row missing-value count, a voltage-current power proxy, and a temperature-duration proxy.",
        "<b>Invalid-record signal engineering.</b> Two leak-free, label-free anomaly features were added before classification: (1) the disagreement between two independently trained regressors (ExtraTrees vs RandomForest) on the predicted Reference_Parameter, computed out-of-fold for training rows and directly for test rows -- rows with inconsistent sensor physics tend to produce disagreeing predictions; (2) an Isolation Forest anomaly score over the full engineered feature space.",
        f"<b>Reference prediction.</b> An ExtraTreesRegressor (400 trees, minimum leaf size 2, 90% feature subsampling) was selected using five-fold shuffled cross-validation. Cross-validated MAE was {metrics['regression_mae']:.4f}, RMSE {metrics['regression_rmse']:.4f}, and R2 {metrics['regression_r2']:.4f}.",
        f"<b>Invalid detection.</b> ExtraTreesClassifier and RandomForestClassifier (both with balanced class weighting) were each evaluated with out-of-fold probabilities averaged over {N_CLASSIFICATION_REPEATS} independent stratified five-fold splits, using the anomaly-augmented feature set. For each model the decision threshold was chosen to directly maximize out-of-fold invalid-class F1 (not fixed in advance), and the better-performing model, {metrics['classification_model']}, was selected. This produced accuracy {metrics['classification_accuracy']:.4f}, invalid precision {metrics['invalid_precision']:.4f}, invalid recall {metrics['invalid_recall']:.4f}, and invalid F1 {metrics['invalid_f1']:.4f} at threshold {metrics['classification_threshold']:.3f}.",
        "<b>Validation and assumptions.</b> Regression folds were shuffled K-folds; classification folds were stratified and repeated with different shuffles to stabilize the threshold estimate. Rows are assumed independent (no repeated group key supplied). Extreme values are retained and handled by the ensembles rather than deleted. Reported metrics are validation estimates, not hidden-test performance.",
        "<b>Digital-twin automation.</b> In deployment, equipment sensors feed a timestamped acquisition layer. A validation gateway checks schema, units, ranges, missingness, and duplicate events, then applies the locked preprocessing and feature pipeline (including the disagreement and anomaly-score computations). The digital twin estimates the current operating state, predicts the reference parameter, and assigns a valid/invalid probability. Confidence, drift, and sensor-health monitors route alerts to operators; confirmed outcomes are stored for periodic recalibration, champion/challenger testing, and controlled retraining.",
    ]
    for paragraph in body:
        story.append(Paragraph(paragraph, styles["Small"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Input features: " + ", ".join(features), styles["Small"]))
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=13 * mm, bottomMargin=13 * mm)
    doc.build(story)


def main():
    for directory in [OUTPUT_DIR, REPORT_DIR, SUBMISSION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    frames, train_sheet, test_sheet = identify_sheets(DATASET_PATH)
    train, test = frames[train_sheet].copy(), frames[test_sheet].copy()
    validate_inputs(train, test)
    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET}]
    x_train = add_features(train, feature_columns)
    x_test = add_features(test, feature_columns)
    y_reg = train[REGRESSION_TARGET].astype(float)
    y_class = (train[CLASSIFICATION_TARGET].astype(str).str.strip().str.lower() == "invalid").astype(int)

    x_train, x_test = add_anomaly_features(x_train, x_test, y_reg)
    all_feature_names = list(x_train.columns)

    comparison, regression_model, classifier, classification_threshold, classifier_name = evaluate_models(x_train, y_reg, y_class)
    regression_row = comparison[comparison.task == "regression"].sort_values("mae").iloc[0]
    classification_row = comparison[(comparison.task == "classification") & (comparison.model == classifier_name)].iloc[0]

    regression_model.fit(x_train, y_reg)
    classifier.fit(x_train, y_class)
    predictions = regression_model.predict(x_test)
    invalid_probability = classifier.predict_proba(x_test)[:, 1]
    invalid_prediction = invalid_probability >= classification_threshold
    labels = np.where(invalid_prediction, "Invalid", "Valid")
    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(str), "Predicted_Reference_Parameter": predictions.astype(float), CLASSIFICATION_TARGET: labels})
    if list(submission.columns) != [ID_COLUMN, "Predicted_Reference_Parameter", CLASSIFICATION_TARGET] or len(submission) != len(test) or submission[ID_COLUMN].duplicated().any() or submission.isna().any().any():
        raise ValueError("Submission schema or completeness validation failed")
    submission_path = OUTPUT_DIR / f"{TEAM_NAME}.csv"
    submission.to_csv(submission_path, index=False)
    summary = {
        "team_name": TEAM_NAME,
        "dataset": {"path": str(DATASET_PATH), "training_sheet": train_sheet, "test_sheet": test_sheet, "training_rows": len(train), "test_rows": len(test), "feature_columns": all_feature_names, "missing_values_training": train.isna().sum().to_dict(), "duplicate_rows_training": int(train.duplicated().sum())},
        "validation": {"strategy": "5-fold shuffled KFold for regression; 5-fold stratified KFold repeated 5x (averaged OOF probabilities) for classification", "seed": SEED, "regression_model": regression_row.model, "regression_mae": float(regression_row.mae), "regression_rmse": float(regression_row.rmse), "regression_r2": float(regression_row.r2), "classification_model": classifier_name, "classification_threshold": float(classification_threshold), "classification_accuracy": float(classification_row.accuracy), "invalid_precision": float(classification_row.invalid_precision), "invalid_recall": float(classification_row.invalid_recall), "invalid_f1": float(classification_row.invalid_f1), "confusion_matrix_valid_invalid": classification_row.confusion_matrix_valid_invalid},
        "prediction_summary": {"predicted_valid": int((labels == "Valid").sum()), "predicted_invalid": int((labels == "Invalid").sum()), "reference_min": float(predictions.min()), "reference_max": float(predictions.max()), "reference_mean": float(predictions.mean()), "reference_median": float(np.median(predictions))},
        "submission_schema": list(submission.columns),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "task3_note": "No separate Task 3 specification was present in the workspace; this summary is generated from the verified pipeline outputs and validation results.",
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    numeric_training = train.select_dtypes(include=[np.number])
    eda = {"sheets": {name: {"rows": int(frame.shape[0]), "columns": int(frame.shape[1]), "column_names": list(frame.columns), "missing_values": frame.isna().sum().to_dict(), "duplicate_rows": int(frame.duplicated().sum())} for name, frame in frames.items()}, "training_numeric_summary": numeric_training.describe().round(6).to_dict()}
    (REPORT_DIR / "data_understanding.json").write_text(json.dumps(eda, indent=2, default=str), encoding="utf-8")
    plt.figure(figsize=(7, 4)); train[REGRESSION_TARGET].hist(bins=30, color="#245b70"); plt.title("Training Reference Parameter"); plt.xlabel(REGRESSION_TARGET); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(REPORT_DIR / "reference_parameter_distribution.png", dpi=140); plt.close()
    write_methodology(
        REPORT_DIR / "methodology.pdf",
        {
            "regression_mae": regression_row.mae,
            "regression_rmse": regression_row.rmse,
            "regression_r2": regression_row.r2,
            "classification_model": classifier_name,
            "classification_threshold": classification_threshold,
            "classification_accuracy": classification_row.accuracy,
            "invalid_precision": classification_row.invalid_precision,
            "invalid_recall": classification_row.invalid_recall,
            "invalid_f1": classification_row.invalid_f1,
        },
        train.shape,
        test.shape,
        all_feature_names,
    )
    files_to_copy = [Path("Dataset.xlsx"), Path("run_pipeline.py"), Path("requirements.txt"), Path("README.md"), submission_path, summary_path, REPORT_DIR / "methodology.pdf", REPORT_DIR / "model_comparison.csv", REPORT_DIR / "data_understanding.json"]
    if (REPORT_DIR / "reference_parameter_distribution.png").exists(): files_to_copy.append(REPORT_DIR / "reference_parameter_distribution.png")
    for source in files_to_copy:
        destination = SUBMISSION_DIR / source.name
        if source.is_file(): shutil.copy2(source, destination)
    zip_path = Path(f"{TEAM_NAME}-submission.zip")
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in SUBMISSION_DIR.rglob("*"):
            if source.is_file(): archive.write(source, source.as_posix())
    print(json.dumps({"submission": str(submission_path), "summary": str(summary_path), "zip": str(zip_path), "test_rows": len(test), "predicted_invalid": int((labels == "Invalid").sum()), "regression_model": regression_row.model, "regression_mae": float(regression_row.mae), "classification_model": classifier_name, "classification_accuracy": float(classification_row.accuracy), "classification_f1_invalid": float(classification_row.invalid_f1)}, indent=2))


if __name__ == "__main__":
    main()
