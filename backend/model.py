from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap

# =========================
# Paths & constants
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent  # /app/backend -> /app

MODEL_PATH = BASE_DIR / "model" / "catboost_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "features.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

# =========================
# Load artifacts (fail fast)
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

model: CatBoostClassifier = joblib.load(MODEL_PATH)
features: list[str] = joblib.load(FEATURES_PATH)

# =========================
# Load metadata (optional)
# =========================
metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

MODEL_VERSION = metadata.get("model_version", "unknown")
TRAINING_DATE = metadata.get("training_date", "unknown")
TARGET = metadata.get("target", "unknown")

# =========================
# Input validationModel info for audit
# =========================
def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures correct feature order, type coercion,
    and missing feature handling.
    """
    # Add missing features as NaN
    for f in features:
        if f not in df.columns:
            df[f] = np.nan

    # Enforce column order
    df = df[features]

    # Convert everything to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================
# Prediction interface
# =========================
def predict_proba(data: dict) -> float:
    """
    Predicts mortality probability for a single patient.

    Parameters
    ----------
    data : dict
        Feature-value mapping

    Returns
    -------
    float
        Probability of positive class
    """
    df = pd.DataFrame([data])
    df = validate_input(df)

    proba = model.predict_proba(df)[0, 1]

    # Safety clamp
    return float(np.clip(proba, 0.0, 1.0))

# =========================
# SHAP plot
# =========================

_explainer = None

def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap.Explainer(model)
    return _explainer

def shap_top_features(data: dict, top_n: int = 20) -> dict:
    df = validate_input(pd.DataFrame([data]))
    explainer = get_explainer()
    shap_values = explainer(df)

    values = shap_values.values[0]
    importance = (
        pd.Series(abs(values), index=features)
        .sort_values(ascending=False)
        .head(top_n)
    )

    return importance.to_dict()

# =========================
# Model info for audit
# =========================
def model_info() -> dict:
    return {
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "target": TARGET,
        "n_features": len(features),
        "features": features,
    }
