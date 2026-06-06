"""
Exploratory Data Analysis for Student Performance Dataset.
Generates statistical summaries and visualizations saved to analysis/plots/.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


def load_data() -> pd.DataFrame:
    """Load the student performance dataset from CSV."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("   Run 'python data/generate_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} records from {data_path}")
    return df


def print_basic_stats(df: pd.DataFrame):
    """Display basic statistics of the dataset."""
    print("\n" + "=" * 60)
    print("  BASIC STATISTICS")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Types:\n{df.dtypes.to_string()}")
    print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
    print(f"\nDescriptive Statistics:")
    print(df.describe().round(2).to_string())


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Generate and save a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_distributions(df: pd.DataFrame, output_dir: str):
    """Generate distribution plots for all features."""
    features = df.columns.tolist()
    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    colors = sns.color_palette("viridis", n_features)

    for i, col in enumerate(features):
        ax = axes[i]
        sns.histplot(df[col], kde=True, color=colors[i], ax=ax, edgecolor="white",
                     linewidth=0.5, alpha=0.7)
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_feature_importance(df: pd.DataFrame, output_dir: str):
    """Calculate and plot feature importance using Random Forest."""
    feature_cols = [c for c in df.columns if c != "performance_score"]
    X = df[feature_cols]
    y = df["performance_score"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        importances.index,
        importances.values,
        color=sns.color_palette("magma", len(importances)),
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_title("Feature Importance (Random Forest)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_xlim(0, importances.max() * 1.15)

    # Clean up tick labels
    labels = [label.replace("_", " ").title() for label in importances.index]
    ax.set_yticklabels(labels, fontsize=11)

    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_scatter_vs_target(df: pd.DataFrame, output_dir: str):
    """Scatter plots of each feature vs. performance_score."""
    feature_cols = [c for c in df.columns if c != "performance_score"]
    n = len(feature_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.scatter(df[col], df["performance_score"], alpha=0.3, s=15,
                   color=sns.color_palette("coolwarm", n)[i], edgecolors="none")
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Performance Score", fontsize=10)
        ax.set_title(f"{col.replace('_', ' ').title()} vs Score",
                     fontsize=11, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Features vs Performance Score", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "scatter_vs_target.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def main():
    """Run the full EDA pipeline."""
    print("=" * 60)
    print("  Exploratory Data Analysis — Student Performance")
    print("=" * 60)

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data()

    # Basic statistics
    print_basic_stats(df)

    # Generate plots
    print("\n📈 Generating visualizations...")
    plot_correlation_heatmap(df, output_dir)
    plot_distributions(df, output_dir)
    plot_feature_importance(df, output_dir)
    plot_scatter_vs_target(df, output_dir)

    print(f"\n✅ All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
