from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model import FEATURES, TARGET, MODEL_PATH

DATA_PATH = Path(__file__).with_name("liver_patient.csv")


def _ensure_columns_present(data: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise KeyError(f"Training data missing expected columns: {missing}")


def _prepare_target(series: pd.Series) -> pd.Series:
    """Normalize target values to 0/1 integers.

    Known encodings in example data are '1' and '2' where '1' indicates the
    positive class and '2' the negative; this function maps them to 1 and 0.
    """
    mapping = {"1": 1, "2": 0, 1: 1, 2: 0, "0": 0, 0: 0}
    result = series.map(mapping)
    if result.isna().any():
        # Try a fallback: coerce to integers and map again
        try:
            coerced = series.astype(int).map(mapping)
            if coerced.isna().any():
                raise ValueError("Target column contains unknown values after coercion.")
            return coerced.astype(int)
        except Exception:
            raise ValueError("Could not normalize target values to 0/1."
                             " Inspect the Dataset/target column values.")
    return result.astype(int)


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    # Normalize column names to match the FEATURES contract
    data.columns = [column.strip().replace(" ", "_") for column in data.columns]

    _ensure_columns_present(data)

    # Prepare X and y
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = _prepare_target(data[TARGET])

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(f"Validation accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    # Persist model with light compression
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
