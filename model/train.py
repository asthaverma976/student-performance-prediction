"""
Model Training Script for Student Performance Prediction.
Trains and compares Linear Regression, Random Forest, and Decision Tree models.
Saves the best model and scaler to disk.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_COLS = [
    "study_hours",
    "attendance_percentage",
    "previous_scores",
    "sleep_hours",
    "extracurricular_activities",
    "parent_education_level",
]
TARGET_COL = "performance_score"


def load_dataset() -> pd.DataFrame:
    """Load the student performance dataset."""
    path = os.path.abspath(DATA_PATH)
    if not os.path.exists(path):
        print(f"❌ Dataset not found at {path}")
        print("   Run 'python data/generate_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} records from {path}")
    return df


def train_models():
    """Train, evaluate, and save the best model."""

    print("=" * 60)
    print("  Model Training — Student Performance Prediction")
    print("=" * 60)

    # --- Load data ---
    df = load_dataset()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📦 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Define models ---
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, random_state=42
        ),
    }

    # --- Train & evaluate ---
    results = {}
    print("\n" + "-" * 60)
    print(f"  {'Model':<25s} {'R²':>8s} {'MAE':>8s} {'MSE':>10s}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results[name] = {"model": model, "r2": r2, "mae": mae, "mse": mse}
        print(f"  {name:<25s} {r2:>8.4f} {mae:>8.2f} {mse:>10.2f}")

    print("-" * 60)

    # --- Select best model by R² ---
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_info = results[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   R² = {best_info['r2']:.4f}  |  MAE = {best_info['mae']:.2f}  |  MSE = {best_info['mse']:.2f}")

    # --- Save model and scaler ---
    joblib.dump(best_info["model"], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Scaler saved to: {SCALER_PATH}")
    print("=" * 60)

    return best_info["model"], scaler, results


if __name__ == "__main__":
    train_models()
