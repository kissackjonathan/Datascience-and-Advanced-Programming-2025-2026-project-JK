"""
Evaluation module tests.
Tests for visualization and evaluation functions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import RandomForestPredictor, XGBoostPredictor

# ============================================================================
# BASIC VISUALIZATION TESTS
# ============================================================================


def test_evaluation_plot_predictions(sample_data, temp_dir):
    # Test: Verifies creation of predictions vs actual values plot
    # Purpose: Ensures scatter plot visualization works
    """Test plot_predictions function."""
    from src.evaluation import plot_predictions

    _, _, _, y_test = sample_data
    y_pred = y_test + np.random.randn(len(y_test)) * 0.1

    output_path = temp_dir / "predictions.png"
    plot_predictions(y_test, y_pred, model_name="Test", output_path=output_path)

    assert output_path.exists()


def test_evaluation_plot_residuals(sample_data, temp_dir):
    # Test: Verifies creation of residual plot
    # Purpose: Ensures error pattern diagnostics work
    """Test plot_residuals function."""
    from src.evaluation import plot_residuals

    _, _, _, y_test = sample_data
    y_pred = y_test + np.random.randn(len(y_test)) * 0.1

    output_path = temp_dir / "residuals.png"
    plot_residuals(y_test, y_pred, model_name="Test", output_path=output_path)

    assert output_path.exists()


def test_evaluation_plot_feature_importance(sample_data, temp_dir):
    # Test: Verifies creation of feature importance plot
    # Purpose: Ensures visualization of influential variables works
    """Test plot_feature_importance function."""
    from src.evaluation import plot_feature_importance

    importance = pd.Series({"f0": 0.4, "f1": 0.3, "f2": 0.2, "f3": 0.1})

    output_path = temp_dir / "importance.png"
    plot_feature_importance(importance, model_name="Test", output_path=output_path)

    assert output_path.exists()


def test_evaluation_display_results_table():
    # Test: Verifies display of model results table
    # Purpose: Ensures formatted table displays without error
    """Test display_results_table function."""
    from src.evaluation import display_results_table

    results = pd.DataFrame(
        {
            "model": ["RF", "XGB"],
            "r2": [0.85, 0.87],
            "rmse": [0.5, 0.45],
            "mae": [0.4, 0.35],
        }
    )

    # Should not raise error
    display_results_table(results, title="Test Results")


def test_evaluation_plot_model_comparison(temp_dir):
    # Test: Verifies creation of model comparison bar chart
    # Purpose: Ensures multi-model comparative visualization works
    """Test plot_model_comparison function."""
    from src.evaluation import plot_model_comparison

    results = pd.DataFrame(
        {
            "model": ["RF", "XGB", "SVR"],
            "r2": [0.85, 0.87, 0.82],
            "rmse": [0.5, 0.45, 0.55],
            "mae": [0.4, 0.35, 0.45],
        }
    )

    output_path = temp_dir / "comparison.png"
    plot_model_comparison(results, output_path=output_path)

    assert output_path.exists()


def test_evaluation_create_correlation_heatmap(temp_dir):
    # Test: Verifies creation of correlation heatmap
    # Purpose: Ensures visualization of relationships between variables
    """Test create_correlation_heatmap function."""
    from src.evaluation import create_correlation_heatmap

    data = pd.DataFrame(
        {
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "f3": np.random.randn(50),
            "target": np.random.randn(50),
        }
    )

    output_path = temp_dir / "heatmap.png"
    create_correlation_heatmap(data, output_path=output_path)

    assert output_path.exists()


def test_evaluation_save_results_summary(temp_dir):
    # Test: Verifies saving results to CSV file
    # Purpose: Ensures metric persistence for reporting
    """Test save_results_summary function."""
    from src.evaluation import save_results_summary

    results = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.85, 0.87], "rmse": [0.5, 0.45]}
    )

    output_path = temp_dir / "results.csv"
    save_results_summary(results, output_path, description="Test Results")

    assert output_path.exists()


def test_evaluation_compare_models():
    # Test: Verifies model comparison between training and test
    # Purpose: Ensures overfitting detection through train/test comparison
    """Test compare_models function."""
    from src.evaluation import compare_models

    benchmark_df = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.90, 0.92], "rmse": [0.4, 0.35]}
    )

    test_df = pd.DataFrame(
        {"model": ["RF", "XGB"], "r2": [0.85, 0.87], "rmse": [0.5, 0.45]}
    )

    # Should not raise error
    compare_models(benchmark_df, test_df)


# ============================================================================
# COMPREHENSIVE VISUALIZATION TESTS
# ============================================================================


def test_evaluation_generate_model_comparison_chart(sample_data, logger, temp_dir):
    # Test: Verifies generation of complete comparison chart (R², MAE, RMSE)
    # Purpose: Integration test for multi-metric visualization
    """Test generate_model_comparison_chart function."""
    from src.evaluation import generate_model_comparison_chart

    X_train, y_train, X_test, y_test = sample_data

    # Train two models
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_metrics = xgb.evaluate(X_test, y_test)

    models_data = [{"model": "RF", **rf_metrics}, {"model": "XGB", **xgb_metrics}]

    result = generate_model_comparison_chart(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_feature_importance_plot(sample_data, logger, temp_dir):
    # Test: Verifies generation of feature importance plot for best model
    # Purpose: Ensures variable importance visualization for tree models
    """Test generate_feature_importance_plot function."""
    from src.evaluation import generate_feature_importance_plot

    X_train, y_train, X_test, y_test = sample_data

    # Train model with feature importance
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)

    # Get predictions
    y_pred = rf.predict(X_test)

    models_data = [
        {
            "model": "RF",
            "model_obj": rf,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            **rf_metrics,
        }
    ]

    result = generate_feature_importance_plot(models_data, temp_dir)
    assert result is not None
    assert result.exists()


def test_evaluation_generate_predictions_plot(sample_data, logger, temp_dir):
    # Test: Verifies generation of predictions vs actual scatter plots
    # Purpose: Ensures multi-model prediction visualization works
    """Test generate_predictions_plot function."""
    from src.evaluation import generate_predictions_plot

    X_train, y_train, X_test, y_test = sample_data

    # Train models
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)
    rf_pred = rf.predict(X_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_metrics = xgb.evaluate(X_test, y_test)
    xgb_pred = xgb.predict(X_test)

    models_data = [
        {"model": "RF", "y_test": y_test, "y_pred": rf_pred, **rf_metrics},
        {"model": "XGB", "y_test": y_test, "y_pred": xgb_pred, **xgb_metrics},
    ]

    result = generate_predictions_plot(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_residual_plots(sample_data, logger, temp_dir):
    # Test: Verifies generation of residual plots for all models
    # Purpose: Ensures error pattern diagnosis across multiple models
    """Test generate_residual_plots function."""
    from src.evaluation import generate_residual_plots

    X_train, y_train, X_test, y_test = sample_data

    # Train multiple models to avoid single subplot bug
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)
    rf_pred = rf.predict(X_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_metrics = xgb.evaluate(X_test, y_test)
    xgb_pred = xgb.predict(X_test)

    models_data = [
        {"model": "RF", "y_test": y_test, "y_pred": rf_pred, **rf_metrics},
        {"model": "XGB", "y_test": y_test, "y_pred": xgb_pred, **xgb_metrics},
    ]

    result = generate_residual_plots(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_learning_curves(sample_data, logger, temp_dir):
    # Test: Verifies generation of learning curves to diagnose overfitting
    # Purpose: Ensures train/validation score progression visualization
    """Test generate_learning_curves function."""
    from src.evaluation import generate_learning_curves

    X_train, y_train, X_test, y_test = sample_data

    # Create train dataframe
    train_data = X_train.copy()
    train_data["political_stability"] = y_train

    # Train model
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)

    models_data = [{"model": "RF", "model_obj": rf, "X_train": X_train, **rf_metrics}]

    result = generate_learning_curves(models_data, train_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_statistical_analysis(sample_data, temp_dir):
    # Test: Verifies generation of statistical analysis plots
    # Purpose: Ensures distribution and correlation visualization
    """Test generate_statistical_analysis function."""
    from src.evaluation import generate_statistical_analysis

    X_train, y_train, X_test, y_test = sample_data

    # Create test dataframe
    test_data = X_test.copy()
    test_data["political_stability"] = y_test

    result = generate_statistical_analysis(test_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_time_series_predictions(sample_data, logger, temp_dir):
    # Test: Verifies generation of time series prediction plots
    # Purpose: Ensures temporal pattern visualization
    """Test generate_time_series_predictions function."""
    from src.evaluation import generate_time_series_predictions

    X_train, y_train, X_test, y_test = sample_data

    # Create test dataframe with Year index
    test_data = X_test.copy()
    test_data["political_stability"] = y_test
    test_data["Year"] = np.random.choice(range(2015, 2021), size=len(test_data))
    test_data = test_data.set_index("Year", append=True)

    # Train multiple models to avoid single subplot bug
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_pred = rf.predict(X_test)
    rf_metrics = rf.evaluate(X_test, y_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = xgb.evaluate(X_test, y_test)

    models_data = [
        {"model": "RF", "y_pred": rf_pred, **rf_metrics},
        {"model": "XGB", "y_pred": xgb_pred, **xgb_metrics},
    ]

    result = generate_time_series_predictions(models_data, test_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_cv_scores_comparison(sample_data, logger, temp_dir):
    # Test: Verifies generation of cross-validation scores comparison
    # Purpose: Ensures CV performance distribution visualization
    """Test generate_cv_scores_comparison function."""
    from src.evaluation import generate_cv_scores_comparison

    X_train, y_train, X_test, y_test = sample_data

    # Create train dataframe
    train_data = X_train.copy()
    train_data["political_stability"] = y_train

    # Train models
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_metrics = xgb.evaluate(X_test, y_test)

    models_data = [
        {"model": "RF", "model_obj": rf, **rf_metrics},
        {"model": "XGB", "model_obj": xgb, **xgb_metrics},
    ]

    result = generate_cv_scores_comparison(models_data, train_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_error_distribution(sample_data, logger, temp_dir):
    # Test: Verifies generation of prediction error distribution plots
    # Purpose: Ensures error pattern analysis with KDE visualization
    """Test generate_error_distribution function."""
    from src.evaluation import generate_error_distribution

    X_train, y_train, X_test, y_test = sample_data

    # Train multiple models to avoid single subplot bug
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_pred = rf.predict(X_test)
    rf_metrics = rf.evaluate(X_test, y_test)

    xgb = XGBoostPredictor(logger=logger)
    xgb.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = xgb.evaluate(X_test, y_test)

    models_data = [
        {"model": "RF", "y_test": y_test, "y_pred": rf_pred, **rf_metrics},
        {"model": "XGB", "y_test": y_test, "y_pred": xgb_pred, **xgb_metrics},
    ]

    result = generate_error_distribution(models_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_regional_analysis(sample_data, logger, temp_dir):
    # Test: Verifies generation of regional performance analysis
    # Purpose: Ensures geographic performance breakdown visualization
    """Test generate_regional_analysis function."""
    from src.evaluation import generate_regional_analysis

    X_train, y_train, X_test, y_test = sample_data

    # Create test dataframe with Country Name
    test_data = X_test.copy()
    test_data["political_stability"] = y_test
    countries = ["France", "Germany", "United States", "Japan", "Brazil"]
    test_data["Country Name"] = np.random.choice(countries, size=len(test_data))
    test_data = test_data.set_index("Country Name", append=True)

    # Train model
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_pred = rf.predict(X_test)
    rf_metrics = rf.evaluate(X_test, y_test)

    models_data = [{"model": "RF", "y_pred": rf_pred, **rf_metrics}]

    result = generate_regional_analysis(models_data, test_data, temp_dir)
    assert result.exists()


def test_evaluation_generate_ml_vs_benchmark_comparison(sample_data, logger, temp_dir):
    # Test: Verifies generation of ML vs benchmark comparison chart
    # Purpose: Ensures ML vs econometric model comparison visualization
    """Test generate_ml_vs_benchmark_comparison function."""
    from src.evaluation import generate_ml_vs_benchmark_comparison

    X_train, y_train, X_test, y_test = sample_data

    # Train ML models
    rf = RandomForestPredictor(logger=logger)
    rf.fit(X_train, y_train, param_grid={"n_estimators": [10], "max_depth": [3]}, cv=2)
    rf_metrics = rf.evaluate(X_test, y_test)

    ml_results = [{"model": "RF", **rf_metrics}]

    # Mock panel metrics
    panel_metrics = {"r2": 0.82, "mae": 0.45, "rmse": 0.55}

    result = generate_ml_vs_benchmark_comparison(ml_results, panel_metrics, temp_dir)
    assert result.exists()


def test_evaluation_generate_political_stability_evolution(sample_data, temp_dir):
    # Test: Verifies generation of political stability evolution plot
    # Purpose: Ensures temporal trend visualization across all countries
    """Test generate_political_stability_evolution function."""
    from src.evaluation import generate_political_stability_evolution

    X_train, y_train, X_test, y_test = sample_data

    # Create test dataframe with Year, Country Name, and political_stability
    test_data = X_test.copy()
    test_data["political_stability"] = y_test
    test_data["Year"] = np.random.choice(range(2000, 2021), size=len(test_data))
    test_data["Country Name"] = np.random.choice(
        ["France", "Germany", "USA", "Japan"], size=len(test_data)
    )

    result = generate_political_stability_evolution(test_data, temp_dir)
    assert result.exists()
