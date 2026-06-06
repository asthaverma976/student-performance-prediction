"""
Synthetic Student Performance Data Generator
Generates 1000 realistic student records with correlated features.
"""

import numpy as np
import pandas as pd
import os


def generate_student_data(num_students: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student performance data with realistic correlations.

    Args:
        num_students: Number of student records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing student features and performance scores.
    """
    np.random.seed(seed)

    # --- Generate independent features ---
    study_hours = np.round(np.random.uniform(0, 12, num_students), 1)
    attendance_percentage = np.round(np.random.uniform(0, 100, num_students), 1)
    previous_scores = np.round(np.random.uniform(0, 100, num_students), 1)
    sleep_hours = np.round(np.random.uniform(4, 10, num_students), 1)
    extracurricular_activities = np.random.choice([0, 1], num_students, p=[0.4, 0.6])
    parent_education_level = np.random.choice([0, 1, 2, 3, 4], num_students,
                                               p=[0.10, 0.20, 0.30, 0.25, 0.15])

    # --- Calculate target: performance_score ---
    # Weighted combination of features (simulates real-world relationships)
    performance_score = (
        study_hours * 4.5                    # Study hours have strong positive effect
        + attendance_percentage * 0.3        # Attendance matters
        + previous_scores * 0.25            # Past performance is predictive
        + sleep_hours * 2.0                  # Adequate sleep helps
        + extracurricular_activities * 3.0   # Extracurriculars give a small boost
        + parent_education_level * 1.5       # Parental education has moderate effect
    )

    # Normalize to 0-100 range
    score_min = performance_score.min()
    score_max = performance_score.max()
    performance_score = (performance_score - score_min) / (score_max - score_min) * 100

    # Add realistic Gaussian noise (±5 points)
    noise = np.random.normal(0, 5, num_students)
    performance_score = performance_score + noise

    # Clip to valid range and round
    performance_score = np.clip(np.round(performance_score, 1), 0, 100)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "previous_scores": previous_scores,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": extracurricular_activities,
        "parent_education_level": parent_education_level,
        "performance_score": performance_score,
    })

    return df


def main():
    """Generate and save the student performance dataset."""
    print("=" * 60)
    print("  Student Performance Data Generator")
    print("=" * 60)

    df = generate_student_data(num_students=1000)

    # Ensure data directory exists
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, "students.csv")

    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df)} student records.")
    print(f"📁 Saved to: {output_path}")
    print(f"\n--- Dataset Preview ---")
    print(df.head(10).to_string(index=False))
    print(f"\n--- Summary Statistics ---")
    print(df.describe().round(2).to_string())
    print(f"\n--- Feature Ranges ---")
    for col in df.columns:
        print(f"  {col:30s} | min: {df[col].min():7.1f} | max: {df[col].max():7.1f}")


if __name__ == "__main__":
    main()
