"""
Evaluation and Visualization Functions

This module contains all functions for:
- Displaying model results (benchmarks, test scores)
- Creating visualizations (plots, charts)
- Comparing model performances

"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#4a90c4",
    "tertiary": "#2c5f8d",
    "accent": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "neutral": "#7f7f7f",
}

# Blue gradient palette (light to dark)
BLUE_PALETTE = ["#e3f2fd", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0", "#0d47a1"]

# Blue-Orange diverging palette
DIVERGING_PALETTE = ["#1f77b4", "#ff7f0e"]  # Blue vs Orange


def display_results_table(results_df: pd.DataFrame, title: str = "Model Results"):
    """
    Display model results in a formatted table.

    Args:
        results_df (pd.DataFrame): DataFrame containing model results with columns r2, rmse, mae
        title (str): Table title (default: "Model Results")

    Returns:
        None
    """
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print("=" * 80)
    print(f"{'Rank':<6} {'Model':<25} {'R^2':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 80)

    # Sort models by R^2 descending (best models first)
    results_sorted = results_df.sort_values("r2", ascending=False).reset_index(
        drop=True
    )

    for idx, row in results_sorted.iterrows():
        rank = idx + 1
        model = row["model"] if "model" in row else row.name
        r2 = row["r2"]
        rmse = row.get("rmse", 0)
        mae = row.get("mae", 0)
        print(f"{rank:<6} {model:<25} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}")

    print("=" * 80)


def compare_models(benchmark_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compare training (benchmark) vs test performance for all models.

    Args:
        benchmark_df (pd.DataFrame): Training set results with columns model, r2
        test_df (pd.DataFrame): Test set results with columns model, r2

    Returns:
        None
    """
    print(f"\n{'=' * 80}")
    print(f"{'MODEL COMPARISON: TRAINING VS TEST':^80}")
    print("=" * 80)
    print(f"{'Model':<25} {'Train R^2':>12} {'Test R^2':>12} {'Difference':>12}")
    print("-" * 80)

    # Merge training and test results by model name
    comparison = pd.merge(
        benchmark_df[["model", "r2"]],
        test_df[["model", "r2"]],
        on="model",
        suffixes=("_train", "_test"),
    )

    # Calculate train-test gap (positive = overfitting, negative = underfitting)
    comparison["diff"] = comparison["r2_train"] - comparison["r2_test"]
    comparison = comparison.sort_values("r2_test", ascending=False)

    for _, row in comparison.iterrows():
        print(
            f"{row['model']:<25} {row['r2_train']:>12.4f} {row['r2_test']:>12.4f} {row['diff']:>12.4f}"
        )

    print("=" * 80)

    # Interpretation
    print("\nInterpretation:")
    print("- Difference > 0.1: Potential overfitting (model memorized training data)")
    print("- Difference ~= 0: Good generalization")
    print("- Difference < 0: Underfitting (model too simple)")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
    show_metrics: bool = True,
):
    """
    Create scatter plot of actual vs predicted values.

    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model for plot title
        output_path (Path): Path where plot image will be saved
        show_metrics (bool): Whether to display metrics text box on plot (default: True)

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot - use consistent blue color
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        color=COLORS["primary"],
        edgecolors="k",
        linewidth=0.5,
    )

    # Calculate plot axis range to ensure perfect prediction line spans entire plot
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    # Perfect prediction line (y=x) - use warning red for reference
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLORS["warning"],
        linestyle="--",
        lw=2,
        label="Perfect Prediction",
    )

    # Calculate and display performance metrics
    if show_metrics:
        # Calculate coefficient of determination (proportion of variance explained)
        r2 = r2_score(y_true, y_pred)
        # Calculate root mean squared error (average prediction error magnitude)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Calculate mean absolute error (average absolute prediction error)
        mae = mean_absolute_error(y_true, y_pred)

        # Add metrics text box
        metrics_text = f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            verticalalignment="top",
            fontsize=10,
        )

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{model_name}: Predicted vs Actual", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"OK Saved: {output_path}")


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path
):
    """
    Create residual plot to check for patterns (model diagnostics).

    Args:
        y_true (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model for plot title
        output_path (Path): Path where plot image will be saved

    Returns:
        None
    """
    # Calculate residuals (prediction errors)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted - use consistent blue
    axes[0].scatter(
        y_pred,
        residuals,
        alpha=0.6,
        color=COLORS["primary"],
        edgecolors="k",
        linewidth=0.5,
    )
    axes[0].axhline(y=0, color=COLORS["warning"], linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Residuals distribution - use consistent blue
    axes[1].hist(
        residuals, bins=30, color=COLORS["primary"], edgecolor="black", alpha=0.7
    )
    axes[1].axvline(x=0, color=COLORS["warning"], linestyle="--", lw=2)
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residuals Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"{model_name}: Residual Analysis", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"OK Saved: {output_path}")


def plot_feature_importance(
    importance: pd.Series, model_name: str, output_path: Path, top_n: int = 10
):
    # Plot feature importance for tree-based models
    plt.figure(figsize=(10, 6))

    # Get top N features
    top_features = importance.head(top_n)

    # Horizontal bar chart - use blue gradient palette
    # Lighter blue = less important, Darker blue = more important
    n_features = len(top_features)
    colors = [BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_features)]
    top_features.sort_values().plot(kind="barh", color=colors, edgecolor="black")

    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(
        f"{model_name}: Top {top_n} Feature Importances", fontsize=14, fontweight="bold"
    )
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"OK Saved: {output_path}")


def plot_model_comparison(
    results_df: pd.DataFrame, output_path: Path, metric: str = "r2"
):
    # Create bar chart comparing all models
    plt.figure(figsize=(12, 6))

    # Sort by metric
    results_sorted = results_df.sort_values(metric, ascending=(metric != "r2"))

    # Create bar chart - use blue gradient (lighter to darker = worse to better)
    n_models = len(results_sorted)
    # Reverse palette for descending order (best model = darkest blue)
    blue_gradient = [
        BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_models)
    ]
    bars = plt.bar(
        range(n_models), results_sorted[metric], color=blue_gradient, edgecolor="black"
    )

    # Customize
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f"Model Comparison: {metric.upper()}", fontsize=14, fontweight="bold")
    plt.xticks(
        range(len(results_sorted)), results_sorted["model"], rotation=45, ha="right"
    )
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, results_sorted[metric])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"OK Saved: {output_path}")


def create_correlation_heatmap(
    data: pd.DataFrame, output_path: Path, title: str = "Feature Correlation Matrix"
):
    """
    Create correlation heatmap for features.

    Args:
        data (pd.DataFrame): DataFrame containing features to correlate
        output_path (Path): Path where heatmap image will be saved
        title (str): Heatmap title (default: "Feature Correlation Matrix")

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))

    # Calculate Pearson correlation matrix between all features
    corr = data.corr()

    # Create heatmap - coolwarm is appropriate for correlations (diverging)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"OK Saved: {output_path}")


def save_results_summary(
    results_df: pd.DataFrame, output_path: Path, description: str = ""
):
    """
    Save results to CSV file.

    Args:
        results_df (pd.DataFrame): Results DataFrame to save
        output_path (Path): Path where CSV will be saved (parent directories created automatically)
        description (str): Optional description to print (default: "")

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    if description:
        print(f"OK {description}: {output_path}")
    else:
        print(f"OK Saved: {output_path}")


# ============================================================================
# BENCHMARK COMPARISON VISUALIZATION
# ============================================================================


def generate_ml_vs_benchmark_comparison(
    ml_results: list, panel_metrics: dict, output_dir: Path
) -> Path:
    # Generate comprehensive comparison of ML models vs Dynamic Panel benchmark
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data
    model_names = [r["model"] for r in ml_results] + ["Dynamic Panel (FE)"]
    r2_scores = [r["r2"] for r in ml_results] + [panel_metrics["r2"]]
    mae_scores = [r["mae"] for r in ml_results] + [panel_metrics.get("mae", 0)]
    rmse_scores = [r["rmse"] for r in ml_results] + [panel_metrics["rmse"]]

    # Colors: ML models in primary blue, benchmark in accent orange
    colors = [COLORS["primary"]] * len(ml_results) + [COLORS["accent"]]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: R^2 Comparison
    bars1 = axes[0].barh(
        model_names,
        r2_scores,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[0].set_xlabel("R^2 Score", fontsize=14, fontweight="bold")
    axes[0].set_title(
        "R^2 Score Comparison\nML Models vs Dynamic Panel Benchmark",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")
    axes[0].axvline(
        x=panel_metrics["r2"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Benchmark",
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, r2_scores)):
        is_best = val == max(r2_scores)
        weight = "bold" if is_best else "normal"
        axes[0].text(val, i, f" {val:.4f}", va="center", fontsize=11, fontweight=weight)

    # Highlight best model
    best_idx = r2_scores.index(max(r2_scores))
    bars1[best_idx].set_edgecolor("gold")
    bars1[best_idx].set_linewidth(3)

    # Plot 2: MAE Comparison (lower is better)
    bars2 = axes[1].barh(
        model_names,
        mae_scores,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_xlabel("MAE (lower is better)", fontsize=14, fontweight="bold")
    axes[1].set_title(
        "Mean Absolute Error Comparison\nML Models vs Dynamic Panel Benchmark",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    axes[1].grid(axis="x", alpha=0.3, linestyle="--")

    if panel_metrics.get("mae"):
        axes[1].axvline(
            x=panel_metrics["mae"],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Benchmark",
        )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, mae_scores)):
        is_best = val == min(mae_scores) if val > 0 else False
        weight = "bold" if is_best else "normal"
        axes[1].text(val, i, f" {val:.4f}", va="center", fontsize=11, fontweight=weight)

    # Highlight best model (lowest MAE)
    if mae_scores:
        best_mae_idx = mae_scores.index(min([m for m in mae_scores if m > 0]))
        bars2[best_mae_idx].set_edgecolor("gold")
        bars2[best_mae_idx].set_linewidth(3)

    # Plot 3: RMSE Comparison (lower is better)
    bars3 = axes[2].barh(
        model_names,
        rmse_scores,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[2].set_xlabel("RMSE (lower is better)", fontsize=14, fontweight="bold")
    axes[2].set_title(
        "Root Mean Squared Error Comparison\nML Models vs Dynamic Panel Benchmark",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    axes[2].grid(axis="x", alpha=0.3, linestyle="--")
    axes[2].axvline(
        x=panel_metrics["rmse"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Benchmark",
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, rmse_scores)):
        is_best = val == min(rmse_scores)
        weight = "bold" if is_best else "normal"
        axes[2].text(val, i, f" {val:.4f}", va="center", fontsize=11, fontweight=weight)

    # Highlight best model (lowest RMSE)
    best_rmse_idx = rmse_scores.index(min(rmse_scores))
    bars3[best_rmse_idx].set_edgecolor("gold")
    bars3[best_rmse_idx].set_linewidth(3)

    # Legend removed to avoid overlapping with plots
    plt.tight_layout()

    # Save figure
    output_file = (
        output_dir
        / f'ml_vs_benchmark_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


# ============================================================================
# COMPREHENSIVE VISUALIZATION FUNCTIONS
# ============================================================================


def generate_model_comparison_chart(models_data: list, output_dir: Path):
    # Generate comprehensive model comparison chart (R^2, MAE, RMSE)
    from datetime import datetime

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = [m["model"] for m in models_data]
    r2_scores = [m["r2"] for m in models_data]
    mae_scores = [m["mae"] for m in models_data]
    rmse_scores = [m["rmse"] for m in models_data]

    # R^2 comparison - use primary blue
    axes[0].barh(
        names, r2_scores, color=COLORS["primary"], alpha=0.8, edgecolor="black"
    )
    axes[0].set_xlabel("R^2 Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Model Comparison - R^2", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(r2_scores):
        axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=10, fontweight="bold")

    # MAE comparison - use secondary blue
    axes[1].barh(
        names, mae_scores, color=COLORS["secondary"], alpha=0.8, edgecolor="black"
    )
    axes[1].set_xlabel("MAE", fontsize=12, fontweight="bold")
    axes[1].set_title("Model Comparison - MAE", fontsize=14, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    for i, v in enumerate(mae_scores):
        axes[1].text(v, i, f" {v:.4f}", va="center", fontsize=10, fontweight="bold")

    # RMSE comparison - use tertiary blue
    axes[2].barh(
        names, rmse_scores, color=COLORS["tertiary"], alpha=0.8, edgecolor="black"
    )
    axes[2].set_xlabel("RMSE", fontsize=12, fontweight="bold")
    axes[2].set_title("Model Comparison - RMSE", fontsize=14, fontweight="bold")
    axes[2].grid(axis="x", alpha=0.3)
    for i, v in enumerate(rmse_scores):
        axes[2].text(v, i, f" {v:.4f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    output_file = (
        output_dir / f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_feature_importance_plot(models_data: list, output_dir: Path):
    # Generate feature importance plot for best model
    from datetime import datetime

    # Find best model by R^2
    best_model = max(models_data, key=lambda x: x["r2"])

    # Get feature names from X_test
    X_test = best_model["X_test"]

    # Check if model has feature_importances_
    if hasattr(best_model["model_obj"].model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(10, 6))
        importances = best_model["model_obj"].model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Use blue gradient palette - lighter to darker
        n_features = len(importances)
        colors = [
            BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_features)
        ]
        ax.bar(range(n_features), importances[indices], color=colors, edgecolor="black")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(
            [X_test.columns[i] for i in indices], rotation=45, ha="right"
        )
        ax.set_xlabel("Features", fontsize=12, fontweight="bold")
        ax.set_ylabel("Importance", fontsize=12, fontweight="bold")
        ax.set_title(
            f'Feature Importance - {best_model["model"]} (R^2={best_model["r2"]:.4f})',
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_file = (
            output_dir
            / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file
    else:
        print(
            f"Warning: {best_model['model']} does not have feature_importances_ attribute"
        )
        return None


def generate_predictions_plot(models_data: list, output_dir: Path):
    # Generate predictions vs actual scatter plots for all models
    import math
    from datetime import datetime

    # Calculate subplot grid dynamically based on number of models
    n_models = len(models_data)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors="k", linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(
            f'{model_info["model"]}\nR^2={model_info["r2"]:.4f}',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Predictions vs Actual - All Models", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    output_file = (
        output_dir
        / f'predictions_vs_actual_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_statistical_analysis(test_data: pd.DataFrame, output_dir: Path):
    # Generate statistical analysis plots (distribution + correlation)
    from datetime import datetime

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Distribution of target variable
    axes[0].hist(
        test_data["political_stability"],
        bins=30,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="black",
    )
    axes[0].set_xlabel("Political Stability Score", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Distribution of Political Stability (Test Set)", fontsize=14, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Correlation heatmap
    corr = test_data.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=axes[1],
    )
    axes[1].set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()

    output_file = (
        output_dir
        / f'statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_residual_plots(models_data: list, output_dir: Path):
    # Generate residual plots for all models
    import math
    from datetime import datetime

    # Calculate subplot grid dynamically based on number of models
    n_models = len(models_data)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]
        residuals = y_test - y_pred

        # Residual scatter plot
        ax.scatter(
            y_pred,
            residuals,
            alpha=0.6,
            color=COLORS["primary"],
            edgecolors="k",
            linewidths=0.5,
        )
        ax.axhline(y=0, color=COLORS["warning"], linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Values", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (R^2={model_info["r2"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Residual Plots - Model Diagnostics", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    output_file = (
        output_dir / f'residual_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_learning_curves(
    models_data: list, train_data: pd.DataFrame, output_dir: Path
):
    # Generate learning curves to diagnose overfitting/underfitting
    from datetime import datetime

    from sklearn.model_selection import learning_curve

    X_train = train_data.drop("political_stability", axis=1)
    y_train = train_data["political_stability"]

    # Calculate subplot grid dynamically based on number of models
    import math

    n_models = len(models_data)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_info in enumerate(models_data):
        try:
            ax = axes[idx]
            model_name = model_info["model"]
            model = model_info["model_obj"]

            # Progress indicator
            print(
                f"  -> Computing learning curve for {model_name}... ({idx+1}/{len(models_data)})"
            )

            # Use the same X_train the model was trained with
            X_to_use = model_info.get("X_train", X_train)

            # Calculate learning curve (OPTIMIZED: fewer points and CV folds for speed)
            # NOTE: No random_state to avoid data leakage - CV shuffle handles randomization
            train_sizes, train_scores, val_scores = learning_curve(
                model.model,
                X_to_use,
                y_train,
                cv=3,
                scoring="r2",  # Reduced from 5 to 3 folds
                train_sizes=np.linspace(0.2, 1.0, 5),  # Reduced from 10 to 5 points
                n_jobs=-1,
                shuffle=True,
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Plot learning curve
            ax.plot(
                train_sizes,
                train_mean,
                "o-",
                color=COLORS["primary"],
                label="Training score",
            )
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color=COLORS["primary"],
            )
            ax.plot(
                train_sizes, val_mean, "o-", color=COLORS["success"], label="CV score"
            )
            ax.fill_between(
                train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color=COLORS["success"],
            )

            ax.set_xlabel("Training Set Size", fontsize=10)
            ax.set_ylabel("R^2 Score", fontsize=10)
            ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(
                f"Warning: Could not generate learning curve for {model_name}: {str(e)}"
            )

    # Hide unused subplots
    for idx in range(len(models_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Learning Curves - Overfitting/Underfitting Diagnosis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = (
        output_dir / f'learning_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_time_series_predictions(
    models_data: list, test_data: pd.DataFrame, output_dir: Path
):
    # Generate time series plot showing predictions over time
    from datetime import datetime

    # Get test data with year information
    test_df = test_data.reset_index()
    y_test = test_df["political_stability"]
    years = test_df["Year"]

    # Calculate subplot grid dynamically based on number of models
    import math

    n_models = len(models_data)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_pred = model_info["y_pred"]

        # Aggregate by year (take mean of predictions per year)
        year_data = pd.DataFrame({"Year": years, "Actual": y_test, "Predicted": y_pred})
        year_avg = year_data.groupby("Year").mean()

        # Plot time series
        ax.plot(
            year_avg.index,
            year_avg["Actual"],
            "o-",
            color=COLORS["primary"],
            label="Actual",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            year_avg.index,
            year_avg["Predicted"],
            "s-",
            color=COLORS["accent"],
            label="Predicted",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Political Stability Score", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (R^2={model_info["r2"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Time Series Predictions - Temporal Evolution",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = (
        output_dir
        / f'time_series_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_cv_scores_comparison(
    models_data: list, train_data: pd.DataFrame, output_dir: Path
):
    # Generate cross-validation scores comparison
    from datetime import datetime

    from sklearn.model_selection import cross_val_score

    X_train = train_data.drop("political_stability", axis=1)
    y_train = train_data["political_stability"]

    cv_results = []

    for model_info in models_data:
        try:
            model_name = model_info["model"]
            model = model_info["model_obj"]

            # Use the same X_train the model was trained with
            X_to_use = model_info.get("X_train", X_train)

            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(
                model.model, X_to_use, y_train, cv=5, scoring="r2", n_jobs=-1
            )

            cv_results.append(
                {
                    "name": model_name,
                    "scores": cv_scores,
                    "mean": np.mean(cv_scores),
                    "std": np.std(cv_scores),
                }
            )

        except Exception as e:
            print(f"Warning: Could not compute CV for {model_name}: {str(e)}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Box plot of CV scores
    bp = ax1.boxplot(
        [r["scores"] for r in cv_results],
        labels=[r["name"] for r in cv_results],
        patch_artist=True,
        showmeans=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["secondary"])
        patch.set_alpha(0.7)

    ax1.set_ylabel("R^2 Score", fontsize=12)
    ax1.set_title(
        "Cross-Validation Scores Distribution", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Mean +/- Std bar chart
    names = [r["name"] for r in cv_results]
    means = [r["mean"] for r in cv_results]
    stds = [r["std"] for r in cv_results]

    x_pos = np.arange(len(names))
    ax2.barh(x_pos, means, xerr=stds, capsize=5, alpha=0.7, color=COLORS["primary"])
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("R^2 Score (Mean +/- Std)", fontsize=12)
    ax2.set_title("Cross-Validation Performance", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(
            mean,
            i,
            f" {mean:.4f}+/-{std:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = (
        output_dir
        / f'cv_scores_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_error_distribution(models_data: list, output_dir: Path):
    # Generate prediction error distribution analysis
    import math
    from datetime import datetime

    from scipy import stats

    # Calculate subplot grid dynamically based on number of models
    n_models = len(models_data)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, model_info in enumerate(models_data):
        ax = axes[idx]
        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]
        errors = y_test - y_pred

        # Create histogram with KDE
        ax.hist(
            errors,
            bins=30,
            alpha=0.6,
            color=COLORS["primary"],
            edgecolor="black",
            density=True,
            label="Error Distribution",
        )

        # Add KDE curve (only if variance > 0)
        if np.var(errors) > 1e-10:  # Check for non-zero variance
            kde = stats.gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(
                x_range, kde(x_range), color=COLORS["warning"], linewidth=2, label="KDE"
            )
        else:
            # If all errors are identical, just add a note
            ax.text(
                0.5,
                0.95,
                "Perfect predictions (variance=0)",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
            )

        # Add vertical line at 0
        ax.axvline(
            x=0,
            color=COLORS["success"],
            linestyle="--",
            linewidth=2,
            label="Zero Error",
        )

        # Add mean line
        ax.axvline(
            x=np.mean(errors),
            color=COLORS["accent"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {np.mean(errors):.3f}",
        )

        ax.set_xlabel("Prediction Error", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(
            f'{model_info["model"]} (MAE={model_info["mae"]:.4f})',
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Prediction Error Distribution Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = (
        output_dir
        / f'error_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_regional_analysis(
    models_data: list, test_data: pd.DataFrame, output_dir: Path
):
    # Generate regional/geographic performance analysis
    from datetime import datetime

    # Get test data with country information
    test_df = test_data.reset_index()

    # Define regions
    regions = {
        "Europe": [
            "France",
            "Germany",
            "Italy",
            "Spain",
            "United Kingdom",
            "Poland",
            "Netherlands",
            "Belgium",
            "Greece",
            "Portugal",
            "Sweden",
            "Austria",
            "Switzerland",
            "Norway",
            "Denmark",
            "Finland",
            "Ireland",
        ],
        "Asia": [
            "China",
            "India",
            "Japan",
            "South Korea",
            "Indonesia",
            "Thailand",
            "Malaysia",
            "Singapore",
            "Philippines",
            "Vietnam",
            "Pakistan",
            "Bangladesh",
        ],
        "Americas": [
            "United States",
            "Canada",
            "Brazil",
            "Mexico",
            "Argentina",
            "Chile",
            "Colombia",
            "Peru",
            "Venezuela",
        ],
        "Africa": [
            "South Africa",
            "Nigeria",
            "Egypt",
            "Kenya",
            "Ghana",
            "Ethiopia",
            "Morocco",
            "Algeria",
            "Tunisia",
        ],
        "Middle East": [
            "Saudi Arabia",
            "United Arab Emirates",
            "Israel",
            "Turkey",
            "Iran",
        ],
    }

    # Map countries to regions
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return "Other"

    test_df["Region"] = test_df["Country Name"].apply(get_region)

    # Get best model (highest R^2)
    best_model = max(models_data, key=lambda x: x["r2"])

    # Calculate regional performance
    regional_perf = []
    for region in ["Europe", "Asia", "Americas", "Africa", "Middle East", "Other"]:
        region_mask = test_df["Region"] == region
        if region_mask.sum() > 0:
            y_true_region = test_df.loc[region_mask, "political_stability"]
            # Convert mask to numpy for proper indexing of y_pred numpy array
            y_pred_region = best_model["y_pred"][region_mask.to_numpy()]

            r2 = r2_score(y_true_region, y_pred_region)
            mae = mean_absolute_error(y_true_region, y_pred_region)
            n_samples = region_mask.sum()

            regional_perf.append(
                {"Region": region, "R^2": r2, "MAE": mae, "Samples": n_samples}
            )

    regional_df = pd.DataFrame(regional_perf)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: R^2 by region - use consistent BLUE_PALETTE
    n_regions = len(regional_df)
    region_colors = [
        BLUE_PALETTE[min(i, len(BLUE_PALETTE) - 1)] for i in range(n_regions)
    ]
    ax1.barh(regional_df["Region"], regional_df["R^2"], color=region_colors, alpha=0.7)
    ax1.set_xlabel("R^2 Score", fontsize=12)
    ax1.set_title(
        f'Regional Performance - {best_model["model"]}', fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="x")

    for i, (r2, n) in enumerate(zip(regional_df["R^2"], regional_df["Samples"])):
        ax1.text(
            r2, i, f" {r2:.3f} (n={n})", va="center", fontsize=10, fontweight="bold"
        )

    # Plot 2: MAE by region - use same consistent colors
    ax2.barh(regional_df["Region"], regional_df["MAE"], color=region_colors, alpha=0.7)
    ax2.set_xlabel("Mean Absolute Error", fontsize=12)
    ax2.set_title(
        f'Regional Error - {best_model["model"]}', fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, axis="x")

    for i, mae in enumerate(regional_df["MAE"]):
        ax2.text(mae, i, f" {mae:.3f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    output_file = (
        output_dir / f'regional_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_political_stability_evolution(data: pd.DataFrame, output_dir: Path):
    """
    Generate simple plot showing average political stability over time.
    Calculates: sum of all country scores / number of countries per year.

    Args:
        data: DataFrame with MultiIndex (Country Name, Year) or regular columns
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot
    """
    from datetime import datetime

    # Ensure we have the data in the right format
    df = data.copy()

    # Reset index if it's a MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Check if we have the required columns
    if "Year" not in df.columns or "political_stability" not in df.columns:
        raise ValueError("Data must contain 'Year' and 'political_stability' columns")

    # Calculate SIMPLE average per year: sum(scores) / count(countries)
    yearly_avg = df.groupby("Year").agg({
        "political_stability": ["sum", "count"]
    }).reset_index()

    # Flatten column names
    yearly_avg.columns = ["Year", "sum", "count"]

    # Calculate average: sum / count
    yearly_avg["average"] = yearly_avg["sum"] / yearly_avg["count"]

    # Create simple figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot average line
    ax.plot(
        yearly_avg["Year"],
        yearly_avg["average"],
        color=COLORS["primary"],
        linewidth=3,
        marker="o",
        markersize=8,
        label="Average Political Stability",
    )

    # Add horizontal line at 0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    # Formatting
    ax.set_xlabel("Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Political Stability Index (Average)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Global Political Stability Evolution\nAverage Across All Countries Over Time",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=12, framealpha=0.9)

    # Add statistics box
    min_year = int(yearly_avg["Year"].min())
    max_year = int(yearly_avg["Year"].max())
    avg_value = yearly_avg["average"].mean()
    min_value = yearly_avg["average"].min()
    max_value = yearly_avg["average"].max()

    # Set Y-axis limits to zoom in on the actual data range
    # Add 20% margin above and below for better visibility
    y_range = max_value - min_value
    y_margin = y_range * 0.20
    ax.set_ylim(min_value - y_margin, max_value + y_margin)

    # Add more Y-axis ticks for better readability of small changes
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

    # Find years for min and max
    min_year_val = int(yearly_avg[yearly_avg["average"] == min_value]["Year"].values[0])
    max_year_val = int(yearly_avg[yearly_avg["average"] == max_value]["Year"].values[0])

    # Calculate total change
    first_value = yearly_avg["average"].iloc[0]
    last_value = yearly_avg["average"].iloc[-1]
    total_change = last_value - first_value

    stats_text = f"""Period: {min_year}-{max_year}
• Overall Average: {avg_value:.3f}
• Lowest: {min_value:.3f} (in {min_year_val})
• Highest: {max_value:.3f} (in {max_year_val})
• Total Change: {total_change:+.3f}
• Countries: {int(yearly_avg['count'].mean())} avg per year"""

    ax.text(
        0.02,
        0.75,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        fontsize=11,
        family="monospace",
    )

    plt.tight_layout()

    output_file = (
        output_dir
        / f'political_stability_evolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def generate_regional_stability_evolution(data: pd.DataFrame, output_dir: Path):
    """
    Generate a plot showing the evolution of political stability by region over time.
    Shows one line per region to compare trends across different geographic areas.

    Args:
        data: DataFrame with MultiIndex (Country Name, Year) or regular columns
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot
    """
    from datetime import datetime

    # Ensure we have the data in the right format
    df = data.copy()

    # Reset index if it's a MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Check if we have the required columns
    if "Year" not in df.columns or "political_stability" not in df.columns:
        raise ValueError("Data must contain 'Year' and 'political_stability' columns")

    # Define regions (same as in generate_regional_analysis)
    regions = {
        "Europe": [
            "France", "Germany", "Italy", "Spain", "United Kingdom", "Poland",
            "Netherlands", "Belgium", "Greece", "Portugal", "Sweden", "Austria",
            "Switzerland", "Norway", "Denmark", "Finland", "Ireland",
        ],
        "Asia": [
            "China", "India", "Japan", "South Korea", "Indonesia", "Thailand",
            "Malaysia", "Singapore", "Philippines", "Vietnam", "Pakistan", "Bangladesh",
        ],
        "Americas": [
            "United States", "Canada", "Brazil", "Mexico", "Argentina", "Chile",
            "Colombia", "Peru", "Venezuela",
        ],
        "Africa": [
            "South Africa", "Nigeria", "Egypt", "Kenya", "Ghana", "Ethiopia",
            "Morocco", "Algeria", "Tunisia",
        ],
        "Middle East": [
            "Saudi Arabia", "United Arab Emirates", "Israel", "Turkey", "Iran",
        ],
    }

    # Map countries to regions
    def get_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return "Other"

    if "Country Name" in df.columns:
        df["Region"] = df["Country Name"].apply(get_region)
    else:
        raise ValueError("Data must contain 'Country Name' column")

    # Calculate regional statistics per year
    regional_yearly = (
        df.groupby(["Region", "Year"])["political_stability"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Define colors for each region
    region_colors = {
        "Europe": "#2E86AB",      # Blue
        "Asia": "#A23B72",        # Purple
        "Americas": "#F18F01",    # Orange
        "Africa": "#C73E1D",      # Red
        "Middle East": "#6A994E", # Green
        "Other": "#95A3A4",       # Gray
    }

    region_markers = {
        "Europe": "o",
        "Asia": "s",
        "Americas": "^",
        "Africa": "D",
        "Middle East": "v",
        "Other": "P",
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # ===== TOP PLOT: Regional Means Over Time =====
    region_list = ["Europe", "Asia", "Americas", "Africa", "Middle East", "Other"]

    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region]
        if len(region_data) > 0:
            ax1.plot(
                region_data["Year"],
                region_data["mean"],
                color=region_colors[region],
                linewidth=2.5,
                marker=region_markers[region],
                markersize=6,
                label=f"{region} (n={region_data['count'].iloc[0]:.0f} countries)",
                alpha=0.8,
            )

    # Add horizontal line at 0
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Formatting top plot
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Political Stability Index", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Political Stability Evolution by Region\nAverage Trends Across Geographic Areas",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)

    # ===== BOTTOM PLOT: Regional Standard Deviations =====
    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region]
        if len(region_data) > 0:
            ax2.plot(
                region_data["Year"],
                region_data["std"],
                color=region_colors[region],
                linewidth=2,
                marker=region_markers[region],
                markersize=5,
                label=region,
                alpha=0.7,
            )

    # Formatting bottom plot
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Standard Deviation", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Political Stability Variability by Region\nWithin-Region Standard Deviation Over Time",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=10, framealpha=0.9, ncol=2)

    # Add statistics box on top plot
    min_year = df["Year"].min()
    max_year = df["Year"].max()

    # Calculate changes for each region
    stats_lines = [f"Regional Changes ({int(min_year)}-{int(max_year)}):"]
    for region in region_list:
        region_data = regional_yearly[regional_yearly["Region"] == region].sort_values("Year")
        if len(region_data) >= 2:
            first_val = region_data["mean"].iloc[0]
            last_val = region_data["mean"].iloc[-1]
            change = last_val - first_val
            stats_lines.append(f"• {region}: {change:+.3f}")

    stats_text = "\n".join(stats_lines)

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    output_file = (
        output_dir
        / f'regional_stability_evolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file
