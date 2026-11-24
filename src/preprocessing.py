"""
preprocessing.py - Data Preprocessing and Leakage Prevention Pipeline

This module implements Phase 5: Data Preprocessing & Leakage Prevention for the
Hotel Booking Cancellation Prediction research project.

Key Responsibilities:
1. clean_dataset: Cleans real-world anomalies (ghost bookings, extreme ADR outliers)
   and imputes missing values safely.
2. create_engineered_features: Adds domain-informed engineered features:
   - total_stay_nights (weekend + week nights)
   - total_guests (adults + children + babies)
   - is_family (binary flag for bookings with children/babies)
   - has_special_requests (binary indicator of special requests)
   - has_booking_changes (binary indicator of modification history)
   - has_previous_cancellations (binary indicator of past cancellations)
   - room_type_changed (binary indicator if assigned room != reserved room)
3. split_data: Strictly drops target leakage features ('reservation_status',
   'reservation_status_date') and performs stratified train/test partitioning.
4. build_preprocessor: Builds a Scikit-Learn ColumnTransformer with StandardScaler
   for numerical features and OneHotEncoder for categorical features.
5. prepare_data: End-to-end orchestration pipeline that fits transformations ONLY
   on training data, transforms train and test sets, saves artifacts to data/processed/,
   and returns all processed datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_path(path_str: Union[str, Path]) -> Path:
    """
    Resolves relative file paths gracefully regardless of whether code
    is executed from the project root or from within the notebooks/ directory.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    
    # Try parent directory if executed from a subdirectory (e.g., notebooks/)
    parent_path = Path("..") / path
    if parent_path.exists():
        return parent_path
    
    # Try searching from current directory parents
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / path
        if candidate.exists():
            return candidate

    return path


def resolve_output_dir(path_str: Union[str, Path]) -> Path:
    """
    Resolves an output directory relative to project root if executed
    from a subdirectory such as notebooks/.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    if Path.cwd().name == "notebooks" and (Path("..") / "data").exists():
        return Path("..") / p
    return p


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleans domain anomalies and imputes missing values in the hotel booking dataset.
    
    Cleaning Operations:
    1. Imputes missing 'children' values (4 records) with 0.
    2. Imputes missing 'country' values with 'Unknown'.
    3. Imputes missing 'agent' and 'company' IDs with 0 (indicating direct / individual booking).
    4. Filters out 180 'ghost bookings' where adults + children + babies == 0.
    5. Filters out extreme ADR outliers (adr < 0 and adr > 5000), eliminating data entry typos.
    
    Parameters:
        df: Raw pandas DataFrame.
        verbose: If True, prints a detailed summary of removed records.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_rows = len(df)
    df_clean = df.copy()

    # 1. Missing Value Imputation
    df_clean["children"] = df_clean["children"].fillna(0)
    df_clean["country"] = df_clean["country"].fillna("Unknown")
    df_clean["agent"] = df_clean["agent"].fillna(0)
    df_clean["company"] = df_clean["company"].fillna(0)

    # 2. Filter ghost bookings (reservations with zero total guests)
    total_guests = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    ghost_mask = total_guests == 0
    ghost_count = int(ghost_mask.sum())

    # 3. Filter extreme ADR pricing outliers (negative charges and extreme $5,400 typo)
    adr_outlier_mask = (df_clean["adr"] < 0) | (df_clean["adr"] > 5000)
    adr_outlier_count = int(adr_outlier_mask.sum())

    # Apply filters
    valid_rows_mask = (~ghost_mask) & (~adr_outlier_mask)
    df_clean = df_clean[valid_rows_mask].copy()
    remaining_rows = len(df_clean)
    total_removed = initial_rows - remaining_rows

    if verbose:
        print("=== DATASET CLEANING SUMMARY ===")
        print(f"Initial raw records:       {initial_rows:,}")
        print(f"Ghost bookings removed:    {ghost_count:,} (adults + children + babies == 0)")
        print(f"ADR outliers removed:      {adr_outlier_count:,} (adr < 0 or adr > 5000)")
        print(f"Total records filtered:    {total_removed:,}")
        print(f"Cleaned records remaining: {remaining_rows:,} ({(remaining_rows / initial_rows) * 100:.2f}%)")
        print(f"Remaining missing values:  {df_clean.isnull().sum().sum()}")
        print("================================")

    return df_clean


def create_engineered_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineers domain-informed features based on exploratory data analysis findings:
    
    1. total_stay_nights: stays_in_weekend_nights + stays_in_week_nights
    2. total_guests: adults + children + babies
    3. is_family: Binary flag (1 if children > 0 or babies > 0, else 0)
    4. has_special_requests: Binary flag (1 if total_of_special_requests > 0, else 0)
    5. has_booking_changes: Binary flag (1 if booking_changes > 0, else 0)
    6. has_previous_cancellations: Binary flag (1 if previous_cancellations > 0, else 0)
    7. room_type_changed: Binary flag (1 if reserved_room_type != assigned_room_type, else 0)
    
    Parameters:
        df: Cleaned pandas DataFrame.
        verbose: If True, prints summary of engineered features.
        
    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    df_feat = df.copy()

    # 1. Total stay duration
    df_feat["total_stay_nights"] = (
        df_feat["stays_in_weekend_nights"] + df_feat["stays_in_week_nights"]
    ).astype(int)

    # 2. Total party size
    df_feat["total_guests"] = (
        df_feat["adults"] + df_feat["children"].fillna(0) + df_feat["babies"]
    ).astype(float)

    # 3. Family indicator
    df_feat["is_family"] = (
        (df_feat["children"].fillna(0) > 0) | (df_feat["babies"] > 0)
    ).astype(int)

    # 4. Special requests indicator
    df_feat["has_special_requests"] = (
        df_feat["total_of_special_requests"] > 0
    ).astype(int)

    # 5. Booking changes indicator
    df_feat["has_booking_changes"] = (df_feat["booking_changes"] > 0).astype(int)

    # 6. Previous cancellation indicator
    df_feat["has_previous_cancellations"] = (
        df_feat["previous_cancellations"] > 0
    ).astype(int)

    # 7. Room assignment change indicator (e.g. upgrades, operational reassignments)
    df_feat["room_type_changed"] = (
        df_feat["reserved_room_type"] != df_feat["assigned_room_type"]
    ).astype(int)

    if verbose:
        print("=== FEATURE ENGINEERING SUMMARY ===")
        print("Engineered 7 domain features:")
        print("  - total_stay_nights          (weekend + weekday nights)")
        print("  - total_guests               (adults + children + babies)")
        print(f"  - is_family                  ({df_feat['is_family'].mean()*100:.2f}% family bookings)")
        print(f"  - has_special_requests       ({df_feat['has_special_requests'].mean()*100:.2f}% with requests)")
        print(f"  - has_booking_changes        ({df_feat['has_booking_changes'].mean()*100:.2f}% modified bookings)")
        print(f"  - has_previous_cancellations ({df_feat['has_previous_cancellations'].mean()*100:.2f}% repeat cancelers)")
        print(f"  - room_type_changed          ({df_feat['room_type_changed'].mean()*100:.2f}% room reassignments)")
        print("===================================")

    return df_feat


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates predictors and target, strictly drops target leakage columns,
    and performs stratified train/test partitioning.
    
    STRICT LEAKAGE PREVENTION:
    - 'reservation_status' and 'reservation_status_date' are post-outcome PMS fields.
      They deterministically expose the cancellation outcome after checkout/cancellation,
      which would produce 100% artificial accuracy and catastrophic real-world failure.
      Both columns are permanently dropped here prior to any split or preprocessing.
    
    Parameters:
        df: DataFrame containing features, target, and optional leakage columns.
        target_col: Name of the binary target variable (default 'is_canceled').
        test_size: Proportion of dataset allocated to the test set (default 0.2).
        random_state: Seed for reproducible random sampling (default 42).
        verbose: If True, prints split shapes and target class balance.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Strict target leakage exclusion list
    leakage_cols = ["reservation_status", "reservation_status_date"]
    
    dropped_leakage = [col for col in leakage_cols if col in df.columns]
    cols_to_drop = [target_col] + dropped_leakage

    if verbose and dropped_leakage:
        print("=== TARGET LEAKAGE PREVENTION ===")
        print(f"STRICTLY DROPPED post-outcome leakage columns: {dropped_leakage}")
        print("Reason: Post-outcome PMS fields deterministically encode target outcome.")
        print("=================================")

    # Separate feature matrix X and target vector y
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    y = df[target_col].copy()

    # Stratified train/test split to preserve exact cancellation proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if verbose:
        print("=== DATA PARTITIONING SUMMARY ===")
        print(f"Total Samples: {len(df):,} rows")
        print(f"Train Partition ({(1 - test_size) * 100:.0f}%): {len(X_train):,} samples")
        print(f"Test Partition  ({test_size * 100:.0f}%): {len(X_test):,} samples")
        print(f"Train Cancellation Rate: {y_train.mean() * 100:.2f}% ({y_train.sum():,} canceled)")
        print(f"Test Cancellation Rate:  {y_test.mean() * 100:.2f}% ({y_test.sum():,} canceled)")
        print(f"Feature Matrix Shape (X): {X.shape[1]} features")
        print("=================================")

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    categorical_features: List[str],
    numerical_features: List[str],
) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for automated feature preprocessing:
    
    - Numerical Pipeline: StandardScaler (z-score standardization: (x - mu) / sigma)
    - Categorical Pipeline: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
      (converts nominal categories to binary dummy features, robust to unseen test levels)
      
    Parameters:
        categorical_features: List of categorical feature column names.
        numerical_features: List of numerical feature column names.
        
    Returns:
        ColumnTransformer: Configured preprocessor object (un-fitted).
    """
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(
    raw_data_path: Union[str, Path] = "data/raw/hotel_bookings.csv",
    output_dir: Union[str, Path] = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the end-to-end data preprocessing pipeline:
    1. Loads raw dataset from raw_data_path.
    2. Cleans anomalies (ghost bookings, ADR typos, missing values).
    3. Generates domain engineered features.
    4. Drops leakage features and executes stratified train/test split.
    5. Identifies categorical and numerical feature sets.
    6. Fits Scikit-Learn preprocessor on training data ONLY (zero data leakage).
    7. Transforms train and test sets to standardized, one-hot encoded matrices.
    8. Exports processed arrays, targets, preprocessor, and feature names to output_dir.
    
    Parameters:
        raw_data_path: File path to the raw hotel_bookings.csv.
        output_dir: Directory to save processed arrays and preprocessor artifacts.
        test_size: Proportion of test split (default 0.2).
        random_state: Random state for reproducibility (default 42).
        verbose: If True, prints progress and verification logs.
        
    Returns:
        Dict[str, Any]: Dictionary containing processed arrays, targets, preprocessor,
                        feature names, and raw data splits.
    """
    raw_path = resolve_path(raw_data_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path} (resolved: {raw_path})")

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading raw dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)

    # 1. Clean anomalies and impute missing values
    clean_df = clean_dataset(raw_df, verbose=verbose)

    # 2. Engineer domain features
    feat_df = create_engineered_features(clean_df, verbose=verbose)

    # 3. Split data and drop target leakage columns
    X_train, X_test, y_train, y_test = split_data(
        feat_df,
        target_col="is_canceled",
        test_size=test_size,
        random_state=random_state,
        verbose=verbose,
    )

    # 4. Identify categorical and numerical columns
    categorical_features = (
        X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    )
    numerical_features = (
        X_train.select_dtypes(include=[np.number]).columns.tolist()
    )

    if verbose:
        print(f"Feature count: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

    # 5. Build preprocessor
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    # 6. Fit preprocessor on training data ONLY (leakage prevention)
    if verbose:
        print("Fitting ColumnTransformer on X_train ONLY...")
    X_train_processed = preprocessor.fit_transform(X_train)

    if verbose:
        print("Transforming X_test using fitted preprocessor...")
    X_test_processed = preprocessor.transform(X_test)

    # Extract clean feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    if verbose:
        print(f"X_train_processed shape: {X_train_processed.shape}")
        print(f"X_test_processed shape:  {X_test_processed.shape}")
        print(f"Total encoded features:  {len(feature_names)}")

    # 7. Save processed artifacts
    if verbose:
        print(f"Saving processed artifacts to: {out_dir}")

    # A. Save NumPy arrays (.npz)
    np.savez_compressed(
        out_dir / "X_train.npz",
        data=X_train_processed.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "X_test.npz",
        data=X_test_processed.astype(np.float32),
    )

    # B. Save Targets (.csv)
    y_train.to_csv(out_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(out_dir / "y_test.csv", index=False, header=True)

    # C. Save Preprocessor (.joblib)
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    # D. Save Feature Names (.json)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # E. Save Parquet if pyarrow / fastparquet is available
    try:
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        X_train_df.to_parquet(out_dir / "X_train.parquet", index=False)
        X_test_df.to_parquet(out_dir / "X_test.parquet", index=False)
        if verbose:
            print("Successfully saved Parquet datasets (X_train.parquet, X_test.parquet).")
    except Exception as e:
        if verbose:
            print(f"Parquet export skipped ({e}). npz format successfully saved.")

    # F. Save unencoded raw splits for tree-based algorithms (LightGBM, CatBoost)
    X_train.to_csv(out_dir / "X_train_raw.csv", index=False)
    X_test.to_csv(out_dir / "X_test_raw.csv", index=False)

    if verbose:
        print("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
        print(f"Artifacts saved in {out_dir}:")
        for p in sorted(out_dir.glob("*")):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {p.name} ({size_mb:.2f} MB)")
        print("============================================")

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "X_train_df": X_train,
        "X_test_df": X_test,
    }


if __name__ == "__main__":
    prepare_data()
