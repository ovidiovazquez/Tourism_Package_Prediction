
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "tourism.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():

    # Load original dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Create a copy of the original dataset
    df_clean = df.copy()

    # Remove unnecessary identifier columns
    df_clean = df_clean.drop(
        columns=["Unnamed: 0", "CustomerID"]
    )

    # Standardize categorical labels
    df_clean["Gender"] = df_clean["Gender"].replace({
        "Fe Male": "Female"
    })

    df_clean["MaritalStatus"] = (
        df_clean["MaritalStatus"].replace({
            "Unmarried": "Single"
        })
    )

    # Separate predictive features and target
    X = df_clean.drop(columns=["ProdTaken"])
    y = df_clean["ProdTaken"]

    # Create group identifiers
    groups = pd.util.hash_pandas_object(
        X,
        index=False
    )

    # Configure stratified group-aware splitting
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Generate candidate splits
    splits = list(
        splitter.split(X, y, groups)
    )

    # Select split closest to 20% test data
    train_idx, test_idx = min(
        splits,
        key=lambda split: abs(
            len(split[1]) / len(X) - 0.20
        )
    )

    # Create training and test datasets
    train_df = df_clean.iloc[train_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    # Keep the target variable as the last column
    feature_columns = X.columns.tolist()
    column_order = feature_columns + ["ProdTaken"]

    train_df = train_df[column_order]
    test_df = test_df[column_order]

    # Verify that identical feature groups do not overlap
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    assert train_groups.isdisjoint(test_groups), (
        "Identical feature groups detected across datasets."
    )

    # Verify total number of records
    assert len(train_df) + len(test_df) == len(df_clean)

    # Save prepared datasets
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        TRAIN_DATA_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    # Display preparation results
    print("\\nTraining dataset:", train_df.shape)
    print("Test dataset:", test_df.shape)

    print("\\nTraining target distribution (%):")
    print(
        train_df["ProdTaken"].value_counts(
            normalize=True
        ) * 100
    )

    print("\\nTest target distribution (%):")
    print(
        test_df["ProdTaken"].value_counts(
            normalize=True
        ) * 100
    )

    print(
        "\\nData preparation completed successfully."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    prepare_data()
