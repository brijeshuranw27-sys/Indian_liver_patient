from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

FEATURES = [
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
TARGET = "Dataset"
MODEL_PATH = Path(__file__).with_name("liver_patient_model.pkl")


def load_model(path: Path = MODEL_PATH) -> Any:
    """Load the trained model from disk.

    Raises FileNotFoundError with a clear instruction if the model file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run 'python train_model.py' to create it."
        )
    return joblib.load(path)


def _to_numeric_dict(payload: Dict[str, Any]) -> Dict[str, float]:
    """Convert and validate payload values to floats for the expected FEATURES.

    Raises ValueError with details on the offending key when conversion fails.
    """
    values: Dict[str, float] = {}
    for feature in FEATURES:
        if feature not in payload:
            raise KeyError(f"Missing feature: {feature}")
        raw = payload[feature]
        try:
            # Accept numeric types or numeric strings
            values[feature] = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{feature}' must be numeric (got: {raw})")
    return values


def predict_patient(payload: Dict[str, Any], model: Optional[Any] = None) -> Dict[str, Any]:
    """Predict using the trained model and return a structured result.

    Result contains:
      - prediction: integer class (0/1)
      - label: human-friendly interpretation
      - probability: probability for the positive class if available (rounded to 4 d.p.), else None
      - features: the numeric feature vector used

    Raises FileNotFoundError if the model artifact is missing.
    """
    values = _to_numeric_dict(payload)
    frame = pd.DataFrame([values], columns=FEATURES)
    model = model or load_model()

    prediction = int(model.predict(frame)[0])

    probability: Optional[float]
    if hasattr(model, "predict_proba"):
        # probability for class '1' (assumes training encoded the positive class as 1)
        probability = float(model.predict_proba(frame)[0][1])
        probability = round(probability, 4)
    else:
        probability = None

    return {
        "prediction": prediction,
        "label": "Likely liver disease" if prediction == 1 else "Lower risk profile",
        "probability": probability,
        "features": values,
    }
