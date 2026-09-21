
import json
import time
import joblib
import mlflow
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MLFLOW_DIR = BASE_DIR / "mlflow"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = (
    MODELS_DIR / "tourism_random_forest_pipeline.joblib"
)

MODEL_SCHEMA_PATH = MODELS_DIR / "model_schema.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
CV_RESULTS_PATH = OUTPUTS_DIR / "random_forest_cv_results.csv"


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model():

    # Create output directories
    for directory in [MODELS_DIR, OUTPUTS_DIR, MLFLOW_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load prepared datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]

    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    assert X_train.columns.tolist() == X_test.columns.tolist()

    print("Training dataset:", train_df.shape)
    print("Test dataset:", test_df.shape)

    # Identify feature types
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    # Define Random Forest
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Build model pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Define the same hyperparameter search space
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
            None
        ]
    }

    # Group identical training feature combinations
    train_groups = pd.util.hash_pandas_object(
        X_train,
        index=False
    )

    cv_strategy = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=2,
        verbose=1,
        refit=True,
        return_train_score=True
    )

    # Configure MLflow with a local SQLite database
    MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    )

    mlflow.set_experiment(
        "Tourism_Package_Prediction_Automated"
    )

    # Execute hyperparameter optimization
    start_time = time.time()

    random_search.fit(
        X_train,
        y_train,
        groups=train_groups
    )

    training_time = time.time() - start_time

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print("Best CV F1-score:", round(best_cv_score, 4))
    print("Selected hyperparameters:", best_params)

    # Save detailed cross-validation results
    cv_results = pd.DataFrame(random_search.cv_results_)

    cv_results.to_csv(
        CV_RESULTS_PATH,
        index=False
    )

    # Log all evaluated configurations
    for index, row in cv_results.iterrows():

        with mlflow.start_run(
            run_name=f"RandomForest_Candidate_{index + 1}"
        ):

            mlflow.log_params(row["params"])

            mlflow.log_metric(
                "mean_cv_f1",
                float(row["mean_test_score"])
            )

            mlflow.log_metric(
                "std_cv_f1",
                float(row["std_test_score"])
            )

            mlflow.log_metric(
                "mean_train_f1",
                float(row["mean_train_score"])
            )

            mlflow.log_metric(
                "mean_fit_time",
                float(row["mean_fit_time"])
            )

            mlflow.log_metric(
                "cv_rank",
                int(row["rank_test_score"])
            )

    # Evaluate selected model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_proba)
        )
    }

    # Register selected model experiment
    with mlflow.start_run(
        run_name="RandomForest_Best_Model"
    ):

        mlflow.log_params(best_params)

        mlflow.log_metric(
            "best_cv_f1",
            float(best_cv_score)
        )

        mlflow.log_metric(
            "training_time_seconds",
            float(training_time)
        )

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        mlflow.set_tag(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.set_tag(
            "cross_validation",
            "StratifiedGroupKFold"
        )

        # Save and register the trained pipeline
        joblib.dump(best_model, MODEL_PATH)

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model"
        )

    # Generate input schema
    model_schema = {
        "model_name": "Tourism Package Prediction",
        "model_type": "RandomForestClassifier",
        "target_variable": "ProdTaken",
        "target_classes": {
            "0": "Not Purchased",
            "1": "Purchased"
        },
        "features": X_train.columns.tolist(),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "categorical_options": {
            feature: sorted(
                X_train[feature].dropna().unique().tolist()
            )
            for feature in categorical_features
        }
    }

    MODEL_SCHEMA_PATH.write_text(
        json.dumps(model_schema, indent=4),
        encoding="utf-8"
    )

    # Save evaluation metrics
    evaluation_results = {
        "model": "RandomForestClassifier",
        "training_records": len(train_df),
        "test_records": len(test_df),
        "best_cv_f1": float(best_cv_score),
        "best_parameters": best_params,
        "training_time_seconds": float(training_time),
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ).tolist()
    }

    METRICS_PATH.write_text(
        json.dumps(evaluation_results, indent=4),
        encoding="utf-8"
    )

    # Verify generated artifacts
    assert MODEL_PATH.exists()
    assert MODEL_SCHEMA_PATH.exists()
    assert METRICS_PATH.exists()
    assert CV_RESULTS_PATH.exists()
    assert MLFLOW_DB_PATH.exists()

    print("\nAutomated model training completed.")

    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nModel artifact:", MODEL_PATH)
    print("Model schema:", MODEL_SCHEMA_PATH)
    print("Evaluation results:", METRICS_PATH)
    print("MLflow database:", MLFLOW_DB_PATH)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()
