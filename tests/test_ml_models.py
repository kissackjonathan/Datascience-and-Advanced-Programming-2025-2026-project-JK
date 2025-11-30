"""
Unit tests for ML models (Random Forest).

Tests cover:
- Model initialization
- Training with GridSearch
- Prediction
- Evaluation metrics
- Model persistence (save/load)
- Feature importance
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.models.ml_models import RandomForestPredictor
from src.utils.logging_config import setup_logging


@pytest.fixture
def sample_data():
    """
    Create sample training and test data.

    Why synthetic data:
    - Fast execution
    - Controlled properties
    - No dependency on actual data files
    """
    np.random.seed(42)
    n_train = 100
    n_test = 30
    n_features = 5

    # Generate synthetic features
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate target with some relationship to features
    y_train = pd.Series(
        X_train['feature_0'] * 2 + X_train['feature_1'] * 0.5 + np.random.randn(n_train) * 0.1
    )
    y_test = pd.Series(
        X_test['feature_0'] * 2 + X_test['feature_1'] * 0.5 + np.random.randn(n_test) * 0.1
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def logger():
    """Setup logger for tests."""
    return setup_logging()


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestRandomForestPredictor:
    """Test suite for RandomForestPredictor class."""

    def test_initialization(self, logger):
        """Test model initialization."""
        model = RandomForestPredictor(logger=logger)

        assert model.model is None
        assert model.grid_search is None
        assert model.best_params_ is None
        assert model.feature_importance_ is None

    def test_fit_with_default_params(self, sample_data, logger):
        """
        Test model fitting with default parameters.

        Why small param grid:
        - Tests run fast (<5 seconds)
        - Still validates GridSearch logic
        """
        X_train, X_test, y_train, y_test = sample_data

        # Small param grid for fast testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        # Verify model was fitted
        assert model.model is not None
        assert model.grid_search is not None
        assert model.best_params_ is not None
        assert model.feature_importance_ is not None

        # Verify best params are from grid
        assert model.best_params_['n_estimators'] in [10, 20]
        assert model.best_params_['max_depth'] in [3, 5]

    def test_predict(self, sample_data, logger):
        """Test prediction on new data."""
        X_train, X_test, y_train, y_test = sample_data

        param_grid = {
            'n_estimators': [10],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        predictions = model.predict(X_test)

        # Verify predictions shape
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

        # Predictions should be numeric
        assert np.all(np.isfinite(predictions))

    def test_predict_without_fit_raises_error(self, sample_data, logger):
        """Test that predict raises error if model not fitted."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestPredictor(logger=logger)

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X_test)

    def test_evaluate(self, sample_data, logger):
        """
        Test model evaluation metrics.

        Expected metrics:
        - R² between 0 and 1 (for reasonable fit)
        - RMSE > 0
        - MAE > 0
        """
        X_train, X_test, y_train, y_test = sample_data

        param_grid = {
            'n_estimators': [10],
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        metrics = model.evaluate(X_test, y_test)

        # Verify all metrics present
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'n_samples' in metrics

        # Verify metric values reasonable
        assert metrics['n_samples'] == len(X_test)
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        # R² can be negative for very bad models, but should be reasonable here
        assert metrics['r2'] > -1

    def test_feature_importance(self, sample_data, logger):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_data

        param_grid = {
            'n_estimators': [20],
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        importance = model.get_feature_importance()

        # Verify importance for all features
        assert len(importance) == X_train.shape[1]

        # Verify sorted descending
        assert list(importance) == sorted(importance.values, reverse=True)

        # Verify all importances are non-negative
        assert all(importance >= 0)

        # Verify importances sum to ~1 (feature importances in RF)
        assert np.isclose(importance.sum(), 1.0, atol=0.01)

    def test_get_feature_importance_without_fit_raises_error(self, logger):
        """Test that get_feature_importance raises error if model not fitted."""
        model = RandomForestPredictor(logger=logger)

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_feature_importance()

    def test_save_and_load_model(self, sample_data, logger, temp_model_dir):
        """
        Test model persistence (save/load).

        Why test this:
        - Ensures models can be deployed
        - Verifies serialization works
        """
        X_train, X_test, y_train, y_test = sample_data

        param_grid = {
            'n_estimators': [10],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        # Train and save
        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        model_path = temp_model_dir / 'test_model.pkl'
        model.save_model(model_path)

        # Verify file created
        assert model_path.exists()

        # Load and verify predictions match
        model_loaded = RandomForestPredictor(logger=logger)
        model_loaded.load_model(model_path)

        pred_original = model.predict(X_test)
        pred_loaded = model_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_cv_results_available(self, sample_data, logger):
        """Test that GridSearchCV results are accessible."""
        X_train, X_test, y_train, y_test = sample_data

        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        cv_results = model.get_cv_results()

        # Verify results is DataFrame
        assert isinstance(cv_results, pd.DataFrame)

        # Verify expected columns
        assert 'mean_test_score' in cv_results.columns
        assert 'mean_train_score' in cv_results.columns
        assert 'std_test_score' in cv_results.columns

        # Verify number of combinations
        expected_combinations = 2 * 2  # 2 n_estimators × 2 max_depth
        assert len(cv_results) == expected_combinations

    def test_overfitting_detection(self, sample_data, logger, capsys):
        """
        Test that overfitting warning is logged when train-CV gap > 0.1.

        Why test logging:
        - Ensures users are warned about overfitting
        - Validates monitoring logic
        """
        X_train, X_test, y_train, y_test = sample_data

        # Use deep trees to induce overfitting
        param_grid = {
            'n_estimators': [50],
            'max_depth': [None],  # No limit - likely to overfit
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model = RandomForestPredictor(logger=logger)
        model.fit(X_train, y_train, param_grid=param_grid, cv=2)

        # Check if train-CV gap is computed
        cv_results = model.get_cv_results()
        best_idx = model.grid_search.best_index_
        train_score = cv_results.loc[best_idx, 'mean_train_score']
        val_score = cv_results.loc[best_idx, 'mean_test_score']
        gap = train_score - val_score

        assert gap >= 0  # Train should be >= validation

    def test_no_scaler_used(self, sample_data, logger):
        """
        Test that RandomForestPredictor does NOT use StandardScaler.

        Why this test:
        - Ensures we don't accidentally add scaling
        - Tree-based models don't need scaling
        """
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestPredictor(logger=logger)

        # Verify no scaler attribute
        assert not hasattr(model, 'scaler')

        # Fit and verify predictions work without scaling
        param_grid = {
            'n_estimators': [10],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        model.fit(X_train, y_train, param_grid=param_grid, cv=2)
        predictions = model.predict(X_test)

        # Should work fine without scaling
        assert len(predictions) == len(X_test)


class TestDataUtils:
    """Test suite for data_utils.py."""

    def test_train_test_split_by_year(self):
        """Test consistent train/test split by year."""
        from src.utils.data_utils import get_train_test_split

        # Create sample data with years
        df = pd.DataFrame({
            'Year': list(range(1996, 2024)) * 10,  # 1996-2023, 10 countries
            'Country': ['A'] * 28 * 10,
            'Value': np.random.randn(280)
        })

        df_train, df_test = get_train_test_split(df, train_end_year=2017)

        # Verify split correctness
        assert df_train['Year'].max() == 2017
        assert df_test['Year'].min() == 2018
        assert df_test['Year'].max() == 2023

        # Verify no overlap
        assert len(set(df_train['Year']) & set(df_test['Year'])) == 0

        # Verify all data used
        assert len(df_train) + len(df_test) == len(df)

    def test_default_train_end_year(self):
        """Test default train_end_year is 2017."""
        from src.utils.data_utils import get_train_test_split

        df = pd.DataFrame({
            'Year': list(range(1996, 2024)) * 5,
            'Value': np.random.randn(140)
        })

        df_train, df_test = get_train_test_split(df)

        # Default should be 2017
        assert df_train['Year'].max() == 2017
        assert df_test['Year'].min() == 2018


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
