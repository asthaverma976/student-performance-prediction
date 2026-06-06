"""
Flask Web Application — Student Performance Prediction System.
Serves a prediction form and displays results.
Optionally stores predictions in MongoDB.
"""

import os
import sys
from flask import Flask, render_template, request

# Add project root to path so we can import our modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.predict import predict_performance

app = Flask(__name__)

# --- Optional MongoDB ---
mongo = None
try:
    from database.mongo_handler import MongoHandler
    mongo = MongoHandler()
    if not mongo.connected:
        mongo = None
except Exception:
    mongo = None


@app.route("/", methods=["GET"])
def index():
    """Render the prediction input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction form submission."""
    try:
        # Collect input features from the form
        features = {
            "study_hours": float(request.form["study_hours"]),
            "attendance_percentage": float(request.form["attendance_percentage"]),
            "previous_scores": float(request.form["previous_scores"]),
            "sleep_hours": float(request.form["sleep_hours"]),
            "extracurricular_activities": int(request.form["extracurricular_activities"]),
            "parent_education_level": int(request.form["parent_education_level"]),
        }

        # Run prediction
        result = predict_performance(features)

        # Optionally store in MongoDB
        if mongo:
            mongo.insert_prediction(
                input_data=features,
                predicted_score=result["predicted_score"],
                performance_category=result["performance_category"],
            )

        return render_template(
            "result.html",
            score=result["predicted_score"],
            category=result["performance_category"],
            confidence=result["confidence"],
            inputs=features,
        )

    except FileNotFoundError as e:
        return render_template(
            "result.html",
            score=0,
            category="Error",
            confidence="N/A",
            inputs={
                "study_hours": 0, "attendance_percentage": 0,
                "previous_scores": 0, "sleep_hours": 0,
                "extracurricular_activities": 0, "parent_education_level": 0,
            },
            error=str(e),
        )
    except Exception as e:
        return f"<h2>Error</h2><p>{str(e)}</p><a href='/'>Go back</a>", 500


if __name__ == "__main__":
    print("\n[*] Starting Student Performance Prediction System...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
