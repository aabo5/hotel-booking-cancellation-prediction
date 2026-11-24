"""
test_pipeline.py - Automated Testing Suite & Reproducibility Verification

Phase 10: Automated Testing Suite & Reproducibility Verification
Covers:
1. Data Loading & Domain Anomaly Cleaning (src/data_loader.py)
2. Preprocessing, Feature Engineering & Zero Leakage Validation (src/preprocessing.py)
3. Model Evaluation & Performance Metrics (src/evaluation.py)
4. Processed Artifacts Integrity & Consistency
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, clean_anomalies, get_raw_data_path
from src.preprocessing import (
    clean_dataset,
    create_engineered_features,
    split_data,
    build_preprocessor,
)
from src.evaluation import evaluate_model


# =============================================================================
# Pytest Fixtures (Module-scoped for high performance and isolation)
# =============================================================================

@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    """Loads raw dataset once for the test module."""
    return load_raw_data()


@pytest.fixture(scope="module")
def cleaned_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Pre-cleans dataset for downstream feature engineering tests."""
    return clean_dataset(raw_data, verbose=False)


@pytest.fixture(scope="module")
def engineered_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Pre-engineers features for downstream split and transformation tests."""
    return create_engineered_features(cleaned_data, verbose=False)


# =============================================================================
# 1. Test Data Loading (src/data_loader.py)
# =============================================================================

def test_raw_data_loading():
    """
    Verify raw CSV loads, has expected 32 columns and >119,000 rows.
    """
    path = get_raw_data_path()
    assert path.exists(), f"Raw data file not found at expected path: {path}"
    
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame), "Raw data must load as a pandas DataFrame"
    assert df.shape[1] == 32, f"Expected 32 columns, but got {df.shape[1]}"
    assert df.shape[0] > 119000, f"Expected >119,000 rows, but got {df.shape[0]:,}"
    assert df.shape[0] == 119390, f"Expected exactly 119,390 rows, but got {df.shape[0]:,}"
    assert "is_canceled" in df.columns, "Target column 'is_canceled' missing from raw dataset"


def test_anomaly_cleaning():
    """
    Verify ghost bookings (adults+children+babies == 0) and extreme ADR anomalies
    (< 0 and > 5000) are filtered properly.
    """
    # 1. Controlled synthetic verification of domain filtering
    synthetic_df = pd.DataFrame({
        "adults": [2, 0, 1, 1, 0],
        "children": [0, 0, 0, 0, 0],
        "babies": [0, 0, 0, 0, 0],
        "adr": [120.0, 100.0, -6.38, 5400.0, 50.0]
    })
    cleaned_syn = clean_anomalies(synthetic_df, verbose=False)
    assert len(cleaned_syn) == 1, (
        f"Expected exactly 1 valid record from synthetic edge-case dataset, got {len(cleaned_syn)}"
    )
    assert cleaned_syn.iloc[0]["adults"] == 2
    assert cleaned_syn.iloc[0]["adr"] == 120.0

    # 2. Verification on the full dataset using clean_anomalies
    raw_df = load_raw_data()
    cleaned_df = clean_anomalies(raw_df, verbose=False)

    # Verify ghost bookings removal (total guests must be > 0)
    total_guests = (
        cleaned_df["adults"] + cleaned_df["children"].fillna(0) + cleaned_df["babies"]
    )
    assert (total_guests == 0).sum() == 0, "Found ghost bookings with 0 total guests in cleaned data"

    # Verify ADR anomaly removal (no negative rates, no extreme outliers > 1000/5000)
    assert (cleaned_df["adr"] < 0).sum() == 0, "Found negative ADR values in cleaned data"
    assert (cleaned_df["adr"] > 5000).sum() == 0, "Found extreme ADR outlier (> 5000) in cleaned data"

    # Expected cleaned size: exactly 119,208 rows (119,390 initial - 180 ghost - 2 ADR outliers)
    assert len(cleaned_df) == 119208, (
        f"Expected 119,208 cleaned records, got {len(cleaned_df):,}"
    )


# =============================================================================
# 2. Test Preprocessing & Zero Leakage (src/preprocessing.py)
# =============================================================================

def test_clean_dataset(raw_data: pd.DataFrame):
    """
    Verify missing values are imputed properly (0 NaNs remain) and anomalies are filtered.
    """
    # 1. Test imputation and filtering on raw dataset
    df_clean = clean_dataset(raw_data, verbose=False)

    # Verify 0 NaNs remain anywhere in the dataset
    nan_count = df_clean.isnull().sum().sum()
    assert nan_count == 0, f"Expected 0 NaNs remaining after clean_dataset, found {nan_count}"

    # Verify specific columns known to have missing values in raw data are imputed
    assert df_clean["children"].isnull().sum() == 0, "'children' column still contains NaNs"
    assert df_clean["country"].isnull().sum() == 0, "'country' column still contains NaNs"
    assert df_clean["agent"].isnull().sum() == 0, "'agent' column still contains NaNs"
    assert df_clean["company"].isnull().sum() == 0, "'company' column still contains NaNs"
    assert (df_clean["country"] == "Unknown").sum() > 0, "Imputed country label 'Unknown' not found"

    # Verify ghost bookings and ADR outliers filtered
    total_guests = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    assert (total_guests == 0).sum() == 0, "Zero-guest bookings remain in cleaned dataset"
    assert ((df_clean["adr"] < 0) | (df_clean["adr"] > 5000)).sum() == 0, "ADR outliers remain"
    assert len(df_clean) == 119208, f"Expected 119,208 records, got {len(df_clean)}"


def test_feature_engineering(cleaned_data: pd.DataFrame):
    """
    Verify that all 7 engineered features are created:
    - total_stay_nights
    - total_guests
    - is_family
    - has_special_requests
    - has_booking_changes
    - has_previous_cancellations
    - room_type_changed
    """
    df_feat = create_engineered_features(cleaned_data, verbose=False)

    expected_features = [
        "total_stay_nights",
        "total_guests",
        "is_family",
        "has_special_requests",
        "has_booking_changes",
        "has_previous_cancellations",
        "room_type_changed",
    ]

    for feat in expected_features:
        assert feat in df_feat.columns, f"Engineered feature '{feat}' is missing from DataFrame"

    # Verify mathematical and logical correctness on sample records
    sample = df_feat.head(500)

    # total_stay_nights = stays_in_weekend_nights + stays_in_week_nights
    expected_nights = sample["stays_in_weekend_nights"] + sample["stays_in_week_nights"]
    assert (sample["total_stay_nights"] == expected_nights).all(), "Calculation error in total_stay_nights"

    # total_guests = adults + children + babies
    expected_guests = sample["adults"] + sample["children"] + sample["babies"]
    assert (sample["total_guests"] == expected_guests).all(), "Calculation error in total_guests"

    # is_family = 1 if children > 0 or babies > 0, else 0
    expected_family = ((sample["children"] > 0) | (sample["babies"] > 0)).astype(int)
    assert (sample["is_family"] == expected_family).all(), "Calculation error in is_family"

    # has_special_requests = 1 if total_of_special_requests > 0, else 0
    expected_req = (sample["total_of_special_requests"] > 0).astype(int)
    assert (sample["has_special_requests"] == expected_req).all(), "Calculation error in has_special_requests"

    # has_booking_changes = 1 if booking_changes > 0, else 0
    expected_changes = (sample["booking_changes"] > 0).astype(int)
    assert (sample["has_booking_changes"] == expected_changes).all(), "Calculation error in has_booking_changes"

    # has_previous_cancellations = 1 if previous_cancellations > 0, else 0
    expected_prev_canc = (sample["previous_cancellations"] > 0).astype(int)
    assert (sample["has_previous_cancellations"] == expected_prev_canc).all(), "Calculation error in has_previous_cancellations"

    # room_type_changed = 1 if reserved_room_type != assigned_room_type, else 0
    expected_room_chg = (sample["reserved_room_type"] != sample["assigned_room_type"]).astype(int)
    assert (sample["room_type_changed"] == expected_room_chg).all(), "Calculation error in room_type_changed"


def test_target_leakage_prevention(engineered_data: pd.DataFrame):
    """
    Verify that 'reservation_status' and 'reservation_status_date' are strictly
    dropped from the feature matrix and never present in X.
    """
    # Confirm leakage columns were originally present in engineered_data
    assert "reservation_status" in engineered_data.columns, (
        "Prerequisite failed: reservation_status not present in input dataframe"
    )
    assert "reservation_status_date" in engineered_data.columns, (
        "Prerequisite failed: reservation_status_date not present in input dataframe"
    )

    X_train, X_test, y_train, y_test = split_data(
        engineered_data, target_col="is_canceled", test_size=0.2, random_state=42, verbose=False
    )

    leakage_cols = ["reservation_status", "reservation_status_date", "is_canceled"]

    for col in leakage_cols:
        assert col not in X_train.columns, (
            f"TARGET LEAKAGE VIOLATION: '{col}' found in X_train!"
        )
        assert col not in X_test.columns, (
            f"TARGET LEAKAGE VIOLATION: '{col}' found in X_test!"
        )

    # Check that target series is cleanly separated
    assert y_train.name == "is_canceled"
    assert y_test.name == "is_canceled"


def test_stratified_split(engineered_data: pd.DataFrame):
    """
    Verify train/test split maintains exact stratified cancellation proportions (~37.08% in both).
    """
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(
        engineered_data, target_col="is_canceled", test_size=test_size, random_state=42, verbose=False
    )

    # Verify partition dimensions
    total_records = len(engineered_data)
    assert len(X_train) + len(X_test) == total_records
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    train_pct = len(X_train) / total_records
    test_pct = len(X_test) / total_records
    assert abs(train_pct - 0.80) < 0.001, f"Train partition ratio {train_pct:.4f} is not ~0.80"
    assert abs(test_pct - 0.20) < 0.001, f"Test partition ratio {test_pct:.4f} is not ~0.20"

    # Verify stratified cancellation proportion (~37.08%)
    train_cancellation_rate = float(y_train.mean())
    test_cancellation_rate = float(y_test.mean())
    overall_cancellation_rate = float(engineered_data["is_canceled"].mean())

    assert abs(train_cancellation_rate - 0.3708) < 0.005, (
        f"Train cancellation rate {train_cancellation_rate:.4f} diverges from expected ~37.08%"
    )
    assert abs(test_cancellation_rate - 0.3708) < 0.005, (
        f"Test cancellation rate {test_cancellation_rate:.4f} diverges from expected ~37.08%"
    )
    # Stratified difference should be negligible (< 0.01% / 0.0001)
    rate_diff = abs(train_cancellation_rate - test_cancellation_rate)
    assert rate_diff < 0.0005, (
        f"Train and Test cancellation rates differ by {rate_diff:.6f}, violating stratification"
    )
    assert abs(train_cancellation_rate - overall_cancellation_rate) < 0.0005


def test_preprocessor_fit_transform(engineered_data: pd.DataFrame):
    """
    Verify that preprocessor can fit on train and transform test without errors
    and produces expected column count (257 features).
    """
    X_train, X_test, y_train, y_test = split_data(
        engineered_data, target_col="is_canceled", test_size=0.2, random_state=42, verbose=False
    )

    categorical_features = (
        X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    )
    numerical_features = (
        X_train.select_dtypes(include=[np.number]).columns.tolist()
    )

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    # Fit ONLY on X_train to prevent leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform X_test with fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Exact expected feature count: 257 features
    assert X_train_processed.shape[1] == 257, (
        f"Expected 257 processed features in train set, got {X_train_processed.shape[1]}"
    )
    assert X_test_processed.shape[1] == 257, (
        f"Expected 257 processed features in test set, got {X_test_processed.shape[1]}"
    )
    assert len(feature_names) == 257, (
        f"Expected 257 feature names, got {len(feature_names)}"
    )

    # Verify row counts match input partitions
    assert X_train_processed.shape[0] == len(X_train)
    assert X_test_processed.shape[0] == len(X_test)

    # Verify no NaN or Inf values produced by transformations
    assert not np.isnan(X_train_processed).any(), "Processed train matrix contains NaN values"
    assert not np.isnan(X_test_processed).any(), "Processed test matrix contains NaN values"
    assert not np.isinf(X_train_processed).any(), "Processed train matrix contains Inf values"
    assert not np.isinf(X_test_processed).any(), "Processed test matrix contains Inf values"


# =============================================================================
# 3. Test Evaluation Module (src/evaluation.py)
# =============================================================================

def test_evaluate_model():
    """
    Verify that evaluate_model correctly returns accuracy, precision, recall,
    f1, roc_auc, and confusion matrix.
    """
    # Create controlled synthetic binary classification task
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    metrics = evaluate_model(clf, X_test, y_test, model_name="TestClassifier")

    # Required metric keys
    expected_metric_keys = [
        "model",
        "accuracy",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
        "confusion_matrix",
        "tn", "fp", "fn", "tp",
        "train_time_sec",
        "pred_time_sec",
        "y_pred",
        "y_prob",
    ]

    for key in expected_metric_keys:
        assert key in metrics, f"Expected key '{key}' missing from evaluate_model output"

    assert metrics["model"] == "TestClassifier"

    # Metric range bounds [0, 1]
    assert 0.0 <= metrics["accuracy"] <= 1.0, f"Accuracy out of bounds: {metrics['accuracy']}"
    assert 0.0 <= metrics["precision_pos"] <= 1.0, f"Precision out of bounds: {metrics['precision_pos']}"
    assert 0.0 <= metrics["recall_pos"] <= 1.0, f"Recall out of bounds: {metrics['recall_pos']}"
    assert 0.0 <= metrics["f1_pos"] <= 1.0, f"F1 out of bounds: {metrics['f1_pos']}"
    assert 0.0 <= metrics["roc_auc"] <= 1.0, f"ROC-AUC out of bounds: {metrics['roc_auc']}"

    # Confusion matrix structure and values
    cm = metrics["confusion_matrix"]
    assert isinstance(cm, np.ndarray), "confusion_matrix must be a numpy ndarray"
    assert cm.shape == (2, 2), f"confusion_matrix shape must be (2, 2), got {cm.shape}"

    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    assert tn + fp + fn + tp == len(y_test), (
        f"Sum of confusion matrix elements ({tn+fp+fn+tp}) does not match test size ({len(y_test)})"
    )
    assert cm[0, 0] == tn and cm[0, 1] == fp and cm[1, 0] == fn and cm[1, 1] == tp


# =============================================================================
# 4. Test Processed Artifacts Integrity
# =============================================================================

def test_saved_artifacts_exist():
    """
    Verify that data/processed/X_train.parquet, X_test.parquet, y_train.csv,
    y_test.csv, and preprocessor.joblib exist on disk, are non-empty, and load properly.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Mapping of artifact filename to its validation criteria
    artifacts = {
        "X_train.parquet": ("parquet", 257),
        "X_test.parquet": ("parquet", 257),
        "y_train.csv": ("csv", None),
        "y_test.csv": ("csv", None),
        "preprocessor.joblib": ("joblib", None),
    }

    for filename, (file_type, expected_cols) in artifacts.items():
        file_path = processed_dir / filename
        assert file_path.exists(), f"Processed artifact '{filename}' not found at {file_path}"
        assert file_path.stat().st_size > 0, f"Processed artifact '{filename}' is empty (0 bytes)"

        if file_type == "parquet":
            df = pd.read_parquet(file_path)
            assert not df.empty, f"Parquet file '{filename}' contains an empty DataFrame"
            assert df.shape[1] == expected_cols, (
                f"Parquet file '{filename}' has {df.shape[1]} columns, expected {expected_cols}"
            )
            assert df.shape[0] > 0, f"Parquet file '{filename}' has 0 rows"

        elif file_type == "csv":
            df = pd.read_csv(file_path)
            assert not df.empty, f"CSV file '{filename}' contains an empty DataFrame"
            assert "is_canceled" in df.columns, (
                f"Target CSV '{filename}' missing 'is_canceled' column"
            )
            assert df.shape[0] > 0, f"CSV file '{filename}' has 0 rows"

        elif file_type == "joblib":
            preprocessor = joblib.load(file_path)
            assert hasattr(preprocessor, "transform"), (
                f"Saved preprocessor '{filename}' missing 'transform' method"
            )
            assert hasattr(preprocessor, "get_feature_names_out"), (
                f"Saved preprocessor '{filename}' missing 'get_feature_names_out' method"
            )
            feature_names = preprocessor.get_feature_names_out()
            assert len(feature_names) == 257, (
                f"Saved preprocessor feature count {len(feature_names)} != 257"
            )
