"""
Data Validation Script
Performs schema and quality checks on raw data.
This script MUST NOT modify the dataset.
"""

import pandas as pd
import sys
import os

RAW_DATA_PATH = "data/raw/breast_cancer.csv"

def validate_data():
    print(" Starting data validation...")

    # 1. Check file exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f" Raw data file not found: {RAW_DATA_PATH}")

    # 2. Load data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f" Data loaded: shape = {df.shape}")

    # 3. Basic schema checks
    expected_total_columns = 31  # 30 features + 1 target
    expected_target_column = "target"

    if df.shape[1] != expected_total_columns:
        raise ValueError(
            f" Expected {expected_total_columns} columns, found {df.shape[1]}"
        )

    if expected_target_column not in df.columns:
        raise ValueError(" Target column 'target' is missing")

    # 4. Feature count check
    feature_columns = df.drop(columns=[expected_target_column])
    if feature_columns.shape[1] != 30:
        raise ValueError(
            f" Expected 30 feature columns, found {feature_columns.shape[1]}"
        )

    # 5. Missing value check
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        raise ValueError(f" Dataset contains {total_missing} missing values")

    # 6. Target value check
    unique_targets = set(df[expected_target_column].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f" Invalid target values found: {unique_targets}"
        )

    # 7. Duplicate row check (warning only)
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f" Warning: {duplicate_rows} duplicate rows detected")

    # 8. Data type sanity check
    non_numeric_cols = feature_columns.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(
            f" Non-numeric feature columns found: {list(non_numeric_cols)}"
        )

    print(" Data validation PASSED")
    return True


if __name__ == "__main__":
    try:
        validate_data()
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(1)