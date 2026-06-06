"""
Prediction Module for Student Performance Prediction System.
Loads the trained model and scaler, then provides a prediction interface.
"""

import os
import joblib
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Feature order must match training
FEATURE_ORDER = [
    "study_hours",
    "attendance_percentage",
    "previous_scores",
    "sleep_hours",
    "extracurricular_activities",
    "parent_education_level",
]

# Lazy-loaded globals
_model = None
_scaler = None


def _load_artifacts():
    """Load the trained model and scaler from disk (once)."""
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run 'python model/train.py' first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run 'python model/train.py' first."
        )

    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)


def predict_performance(features: dict) -> dict:
    """
    Predict a student's performance score.

    Args:
        features: Dict with keys matching FEATURE_ORDER, e.g.:
            {
                "study_hours": 7.5,
                "attendance_percentage": 85.0,
                "previous_scores": 72.0,
                "sleep_hours": 7.0,
                "extracurricular_activities": 1,
                "parent_education_level": 3,
            }

    Returns:
        Dict with:
            - predicted_score (float): Score between 0 and 100.
            - performance_category (str): "Excellent" / "Good" / "Average" / "Poor".
            - confidence (str): Qualitative confidence level.
    """
    _load_artifacts()

    # Build feature array in correct order
    feature_values = [float(features[f]) for f in FEATURE_ORDER]
    X = np.array(feature_values).reshape(1, -1)

    # Scale and predict
    X_scaled = _scaler.transform(X)
    raw_score = _model.predict(X_scaled)[0]

    # Clamp to 0-100
    predicted_score = round(float(np.clip(raw_score, 0, 100)), 2)

    # Determine category
    if predicted_score >= 85:
        category = "Excellent"
        confidence = "High"
    elif predicted_score >= 70:
        category = "Good"
        confidence = "High"
    elif predicted_score >= 50:
        category = "Average"
        confidence = "Moderate"
    else:
        category = "Poor"
        confidence = "Moderate"

    return {
        "predicted_score": predicted_score,
        "performance_category": category,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # Quick test
    sample = {
        "study_hours": 7.5,
        "attendance_percentage": 85.0,
        "previous_scores": 72.0,
        "sleep_hours": 7.0,
        "extracurricular_activities": 1,
        "parent_education_level": 3,
    }
    result = predict_performance(sample)
    print("Sample Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
