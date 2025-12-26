"""
All model tests: ML models and Panel analyzer.
Tests for RF, XGBoost, GradientBoosting, SVR, KNN, MLP, ElasticNet, and PanelAnalyzer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import (
    ElasticNetPredictor,
    GradientBoostingPredictor,
    KNNPredictor,
    MLPPredictor,
    PanelAnalyzer,
    RandomForestPredictor,
    SVRPredictor,
    XGBoostPredictor,
)

# ============================================================================
# NON-PIPELINE ML MODELS (RF, XGBoost, GradientBoosting)
# ============================================================================


@pytest.mark.parametrize(
    "ModelClass,param_grid",
    [
        (RandomForestPredictor, {"n_estimators": [10, 20], "max_depth": [3, 5]}),
        (
            XGBoostPredictor,
            {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.1]},
        ),
        (
            GradientBoostingPredictor,
            {"n_estimators": [20], "max_depth": [3], "learning_rate": [0.1]},
        ),
    ],
)
class TestNonPipelineModels:
    """Test suite for models without Pipeline (RF, XGBoost, GradientBoosting)."""

    def test_initialization(self, ModelClass, param_grid):
        # Test: Verifies that the model can be created without error
        # Purpose: Ensures the constructor works correctly
        """Test model can be initialized."""
        model = ModelClass()
        assert model is not None
        assert model.model is None
        assert model.grid_search is None

    def test_fit(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies that the model trains with GridSearchCV
        # Purpose: Ensures hyperparameter optimization works
        """Test model fitting with GridSearchCV."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        assert model.model is not None
        assert model.grid_search is not None
        assert model.best_params_ is not None

    def test_predict(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies that the model generates valid predictions
        # Purpose: Ensures inference works after training
        """Test model predictions."""
        X_train, y_train, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_evaluate(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies metric calculation (R², RMSE, MAE)
        # Purpose: Ensures performance evaluation works
        """Test model evaluation metrics."""
        X_train, y_train, X_test, y_test = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        metrics = model.evaluate(X_test, y_test)

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(X_test)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_feature_importance(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies feature importance extraction
        # Purpose: Ensures the model can identify influential variables
        """Test feature importance extraction."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X_train.shape[1]
        assert importance.sum() > 0

    def test_save_load(self, ModelClass, param_grid, sample_data, logger, temp_dir):
        # Test: Verifies model saving and loading
        # Purpose: Ensures reproducibility and persistence of trained models
        """Test model persistence."""
        X_train, y_train, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        pred_before = model.predict(X_test)

        # Save and load
        model_path = temp_dir / f"{ModelClass.__name__}.pkl"
        model.save_model(model_path)
        assert model_path.exists()

        model_loaded = ModelClass(logger=logger)
        model_loaded.load_model(model_path)
        pred_after = model_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_cv_results(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies access to cross-validation results
        # Purpose: Allows analyzing performance of each hyperparameter combination
        """Test cross-validation results."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        cv_results = model.get_cv_results()

        assert isinstance(cv_results, pd.DataFrame)
        assert len(cv_results) > 0

    def test_overfitting_detection(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies calculation of gap between training and validation scores
        # Purpose: Detects overfitting by comparing performances
        """Test overfitting gap calculation."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        assert model.train_score_ is not None
        assert model.cv_score_ is not None
        assert model.overfitting_gap_ is not None
        assert model.overfitting_gap_ >= 0

    def test_prediction_without_fit_raises_error(
        self, ModelClass, param_grid, sample_data, logger
    ):
        # Test: Verifies that prediction without training raises an error
        # Purpose: Prevents accidental use of untrained models
        """Test that prediction without fitting raises ValueError."""
        _, _, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        with pytest.raises(ValueError, match="not fitted|model"):
            model.predict(X_test)


# ============================================================================
# PIPELINE ML MODELS (SVR, KNN, MLP, ElasticNet)
# ============================================================================


@pytest.mark.parametrize(
    "ModelClass,param_grid",
    [
        (SVRPredictor, {"model__C": [1, 10], "model__kernel": ["rbf"]}),
        (KNNPredictor, {"model__n_neighbors": [3, 5]}),
        (
            MLPPredictor,
            {"model__hidden_layer_sizes": [(20,)], "model__max_iter": [100]},
        ),
        (ElasticNetPredictor, {"model__alpha": [0.1, 1.0], "model__l1_ratio": [0.5]}),
    ],
)
class TestPipelineModels:
    """Test suite for models using Pipeline (SVR, KNN, MLP, ElasticNet)."""

    def test_initialization(self, ModelClass, param_grid):
        # Test: Verifies that Pipeline model can be created without error
        # Purpose: Ensures constructor with StandardScaler works
        """Test model can be initialized."""
        model = ModelClass()
        assert model is not None
        assert model.model is None
        assert model.grid_search is None

    def test_fit(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies training with Pipeline and GridSearchCV
        # Purpose: Ensures normalization and optimization work together
        """Test model fitting with GridSearchCV."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        assert model.model is not None
        assert model.grid_search is not None
        assert model.best_params_ is not None

    def test_pipeline_structure(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies that Pipeline contains scaler + model
        # Purpose: Ensures correct automatic preprocessing structure
        """Test that Pipeline structure is correct."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        # Verify Pipeline structure
        assert hasattr(model.model, "named_steps")
        assert "scaler" in model.model.named_steps
        assert "model" in model.model.named_steps

    def test_predict(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies that Pipeline model generates valid predictions
        # Purpose: Ensures scaler and model work together for inference
        """Test model predictions."""
        X_train, y_train, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_evaluate(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies metric calculation for Pipeline models
        # Purpose: Ensures evaluation works with normalized data
        """Test model evaluation metrics."""
        X_train, y_train, X_test, y_test = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        metrics = model.evaluate(X_test, y_test)

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(X_test)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_save_load(self, ModelClass, param_grid, sample_data, logger, temp_dir):
        # Test: Verifies saving and loading of complete Pipeline
        # Purpose: Ensures scaler and model are saved together
        """Test model persistence."""
        X_train, y_train, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        pred_before = model.predict(X_test)

        # Save and load
        model_path = temp_dir / f"{ModelClass.__name__}.pkl"
        model.save_model(model_path)
        assert model_path.exists()

        model_loaded = ModelClass(logger=logger)
        model_loaded.load_model(model_path)
        pred_after = model_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_cv_results(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies access to cross-validation results
        # Purpose: Allows analyzing performance of each hyperparameter combination
        """Test cross-validation results."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        cv_results = model.get_cv_results()

        assert isinstance(cv_results, pd.DataFrame)
        assert len(cv_results) > 0

    def test_overfitting_detection(self, ModelClass, param_grid, sample_data, logger):
        # Test: Verifies calculation of gap between training and validation scores
        # Purpose: Detects overfitting by comparing performances
        """Test overfitting gap calculation."""
        X_train, y_train, _, _ = sample_data
        model = ModelClass(logger=logger)

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        assert model.train_score_ is not None
        assert model.cv_score_ is not None
        assert model.overfitting_gap_ is not None
        assert model.overfitting_gap_ >= 0

    def test_prediction_without_fit_raises_error(
        self, ModelClass, param_grid, sample_data, logger
    ):
        # Test: Verifies that prediction without training raises an error
        # Purpose: Prevents accidental use of untrained models
        """Test that prediction without fitting raises ValueError."""
        _, _, X_test, _ = sample_data
        model = ModelClass(logger=logger)

        with pytest.raises(ValueError, match="not fitted|model"):
            model.predict(X_test)


# ============================================================================
# ELASTIC NET SPECIFIC TESTS
# ============================================================================


def test_elastic_net_coefficients(sample_data, logger):
    # Test: Verifies extraction of Elastic Net regression coefficients
    # Purpose: Allows identifying importance of each variable with regularization
    """Test ElasticNet coefficient extraction."""
    X_train, y_train, _, _ = sample_data
    model = ElasticNetPredictor(logger=logger)

    param_grid = {"model__alpha": [0.1], "model__l1_ratio": [0.5]}
    model.fit(X_train, y_train, param_grid=param_grid, cv=2)

    coefficients = model.get_coefficients()

    assert coefficients is not None
    assert len(coefficients) == X_train.shape[1]
    assert all(fname in coefficients.index for fname in X_train.columns)


# ============================================================================
# PANEL ANALYZER TESTS
# ============================================================================


class TestPanelAnalyzer:
    """Test suite for PanelAnalyzer (Fixed Effects panel model)."""

    def test_initialization(self, panel_data, temp_dir):
        # Test: Verifies that PanelAnalyzer can be created with panel data
        # Purpose: Ensures constructor correctly loads longitudinal data
        """Test PanelAnalyzer can be initialized."""
        # Save panel data to CSV
        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        assert analyzer is not None
        assert analyzer.target == "political_stability"

    def test_fit_dynamic_panel(self, panel_data, temp_dir):
        # Test: Verifies dynamic panel model estimation with fixed effects
        # Purpose: Ensures econometric model with lagged variables works
        """Test fitting dynamic panel model with Fixed Effects."""
        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        # Load data first
        analyzer.load_data()

        # Prepare panel structure
        analyzer.prepare_panel_structure()

        # Fit dynamic panel model
        analyzer.fit_dynamic_panel(lags=1)

        assert analyzer.dynamic_results is not None
        assert hasattr(analyzer, "dynamic_lags_")
        assert hasattr(analyzer, "dynamic_lag_name_")
        assert analyzer.dynamic_lags_ == 1

    def test_evaluate_on_test(self, panel_data, temp_dir):
        # Test: Verifies panel model evaluation on test set
        # Purpose: Ensures out-of-sample predictions work correctly
        """Test evaluation on test set."""
        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()

        # Perform train/test split
        analyzer.train_test_split(train_end_year=2015)

        # Fit on training data
        analyzer.fit_dynamic_panel(lags=1)

        # Evaluate on test set
        metrics = analyzer.evaluate_on_test()

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mse" in metrics
        assert "n_test" in metrics

    def test_persistence_analysis(self, panel_data, temp_dir):
        # Test: Verifies shock persistence calculation (AR coefficient)
        # Purpose: Measures duration of temporal effects on target variable
        """Test persistence analysis (AR coefficient)."""
        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()
        analyzer.fit_dynamic_panel(lags=1)

        persistence = analyzer.analyze_persistence()

        # Always present
        assert isinstance(persistence, dict)
        assert "coefficient" in persistence
        assert "p_value" in persistence
        assert "interpretation" in persistence

        # half_life only exists when 0 < coefficient < 1
        coef = persistence["coefficient"]
        if 0 < coef < 1:
            assert "half_life" in persistence
            assert persistence["half_life"] > 0
        else:
            # No half_life for non-stationary or negative coefficients
            assert "half_life" not in persistence or persistence["half_life"] is None

    def test_diagnostic_tests(self, panel_data, temp_dir):
        # Test: Verifies econometric tests (Hausman, autocorrelation, heteroscedasticity)
        # Purpose: Validates statistical assumptions of panel model
        """Test diagnostic tests (Hausman, Autocorrelation, Heteroscedasticity)."""
        csv_path = temp_dir / "panel_data.csv"
        panel_data.to_csv(csv_path, index=False)

        analyzer = PanelAnalyzer(
            data_path=csv_path,
            target="political_stability",
            predictors=[
                "gdp_per_capita",
                "unemployment",
                "inflation",
                "gdp_growth",
                "effectiveness",
                "rule_of_law",
                "trade",
                "hdi",
            ],
        )

        analyzer.load_data()
        analyzer.prepare_panel_structure()
        analyzer.fit_dynamic_panel(lags=1)

        diagnostics = analyzer.run_diagnostic_tests()

        assert "hausman" in diagnostics
        assert "autocorrelation" in diagnostics
        assert "heteroscedasticity" in diagnostics

        # Hausman test validation
        hausman = diagnostics["hausman"]
        assert "statistic" in hausman
        assert "p_value" in hausman
        assert "conclusion" in hausman
        # Statistic should be non-negative chi-squared value
        assert hausman["statistic"] >= 0
        # P-value should be between 0 and 1
        assert 0 <= hausman["p_value"] <= 1

        # Autocorrelation test validation
        autocorr = diagnostics["autocorrelation"]
        assert "rho" in autocorr
        assert "p_value" in autocorr
        assert "conclusion" in autocorr
        # Rho (AR1 coefficient) should be between -1 and 1
        assert -1 <= autocorr["rho"] <= 1
        # P-value should be between 0 and 1
        assert 0 <= autocorr["p_value"] <= 1

        # Heteroscedasticity test validation
        hetero = diagnostics["heteroscedasticity"]
        assert "lm_stat" in hetero
        assert "lm_pvalue" in hetero
        assert "f_stat" in hetero
        assert "f_pvalue" in hetero
        assert "conclusion" in hetero
        # Statistics should be non-negative
        assert hetero["lm_stat"] >= 0
        assert hetero["f_stat"] >= 0
        # P-values should be between 0 and 1
        assert 0 <= hetero["lm_pvalue"] <= 1
        assert 0 <= hetero["f_pvalue"] <= 1
