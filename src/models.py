"""
Model Definitions for Political Stability Prediction

Models:
- RandomForestPredictor          - XGBoostPredictor
- GradientBoostingPredictor      - ElasticNetPredictor
- SVRPredictor                   - KNNPredictor
- MLPPredictor                   - PanelAnalyzer (Dynamic Panel with FE)

Each model is a standalone class with fit(), predict(), and evaluate() methods.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Panel data & econometric libraries
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant

# REPRODUCIBILITY: Fix all random seeds for consistent results
# This ensures that running the code multiple times produces identical results,
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class RandomForestPredictor:
    # Random Forest Regressor with GridSearch optimization

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize Random Forest predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.feature_importance_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.n_features_ = None  # Store number of features for F-stat
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "RandomForestPredictor":
        # Fit Random Forest with GridSearchCV hyperparameter optimization
        if param_grid is None:
            # Default grid with overfitting prevention
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],  # Prevent too deep trees
                "min_samples_split": [2, 5, 10],  # Require samples before split
                "min_samples_leaf": [1, 2, 4],  # Require samples in leaves
                "max_features": ["sqrt", "log2"],  # Limit features per split
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # GridSearchCV with k-fold CV (NO StandardScaler, NO redundant cross_val_score)
        self.grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_SEED),
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,  # To monitor overfitting
        )

        self.grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        # Log results
        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting by comparing train vs CV scores
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        # F = (R² / p) / ((1 - R²) / (n - p - 1))
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        metrics = {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

        return metrics

    def save_model(self, filepath: Path) -> None:
        # Save trained model to disk
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "RandomForestPredictor":
        # Load trained model from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_feature_importance(self) -> pd.Series:
        # Get feature importance rankings (sorted descending)
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.feature_importance_

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class XGBoostPredictor:
    # XGBoost Regressor with GridSearch optimization

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize XGBoost predictor
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.feature_importance_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "XGBoostPredictor":
        # Fit XGBoost with GridSearchCV hyperparameter optimization
        if param_grid is None:
            # Default grid with reasonable overfitting prevention
            param_grid = {
                "n_estimators": [100, 200],  # Number of boosting rounds
                "max_depth": [3, 5, 7],  # Tree depth (prevent overfitting)
                "learning_rate": [0.03, 0.1],  # Step size (small = less overfitting)
                "subsample": [0.8, 1.0],  # Row sampling (like bagging)
                "colsample_bytree": [0.8, 1.0],  # Column sampling (like max_features)
                "reg_alpha": [0, 0.1],  # L1 regularization
                "reg_lambda": [1, 2],  # L2 regularization
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # GridSearchCV with XGBoost (NO early_stopping)
        self.grid_search = GridSearchCV(
            estimator=XGBRegressor(
                random_state=RANDOM_SEED,
                objective="reg:squarederror",
                tree_method="auto",
            ),
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        # Log results
        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting by comparing train vs CV scores
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        # F = (R² / p) / ((1 - R²) / (n - p - 1))
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        metrics = {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

        return metrics

    def save_model(self, filepath: Path) -> None:
        # Save trained model to disk
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "XGBoostPredictor":
        # Load trained model from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_feature_importance(self) -> pd.Series:
        # Get feature importance rankings (sorted descending)
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.feature_importance_

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class KNNPredictor:
    # K-Nearest Neighbors Regressor with GridSearch and StandardScaler

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize KNN predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "KNNPredictor":
        # Fit KNN with GridSearchCV and StandardScaler
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        if param_grid is None:
            # Pipeline requires 'model__' prefix for model parameters
            param_grid = {
                "model__n_neighbors": [3, 5, 7, 10, 15],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # Pipeline prevents data leakage: scaler is refit in each CV fold
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor())]
        )

        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_

        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

    def save_model(self, filepath: Path) -> None:
        """Save trained model (Pipeline with scaler) to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "KNNPredictor":
        # Load trained model (Pipeline with scaler) from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class SVRPredictor:
    # Support Vector Regression with GridSearch and StandardScaler

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize SVR predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "SVRPredictor":
        # Fit SVR with GridSearchCV and StandardScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        if param_grid is None:
            # Pipeline requires 'model__' prefix for model parameters
            # gamma is critical for RBF kernel performance
            param_grid = {
                "model__C": [0.1, 1, 10, 100],
                "model__epsilon": [0.01, 0.1, 0.2],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],  # RBF kernel parameter
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # Pipeline prevents data leakage: scaler is refit in each CV fold
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVR())])

        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_

        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

    def save_model(self, filepath: Path) -> None:
        """Save trained model (Pipeline with scaler) to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "SVRPredictor":
        # Load trained model (Pipeline with scaler) from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class MLPPredictor:
    # Multi-Layer Perceptron (Neural Network) with GridSearch and StandardScaler

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize MLP predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "MLPPredictor":
        # Fit MLP with GridSearchCV and StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        if param_grid is None:
            # Pipeline requires 'model__' prefix for model parameters
            param_grid = {
                "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
                "model__activation": ["relu", "tanh"],  # Activation function
                "model__alpha": [0.0001, 0.001, 0.01],  # L2 regularization
                "model__learning_rate_init": [0.001, 0.01],
                "model__max_iter": [500],
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # Pipeline prevents data leakage: scaler is refit in each CV fold
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        random_state=RANDOM_SEED,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        )

        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_

        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

    def save_model(self, filepath: Path) -> None:
        """Save trained model (Pipeline with scaler) to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "MLPPredictor":
        # Load trained model (Pipeline with scaler) from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class GradientBoostingPredictor:
    # Gradient Boosting Regressor with GridSearch optimization

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize Gradient Boosting predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.feature_importance_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "GradientBoostingPredictor":
        # Fit Gradient Boosting with GridSearchCV hyperparameter optimization
        if param_grid is None:
            # Default grid optimized for political stability prediction
            # Based on notebook experiments: best = 300 trees, lr=0.01, depth=3
            param_grid = {
                "n_estimators": [200, 300, 400],  # More trees for slow learning
                "learning_rate": [0.01, 0.03, 0.05],  # Slow learning = less overfitting
                "max_depth": [3, 4, 5],  # Shallow trees
                "subsample": [0.8, 0.9, 1.0],  # Stochastic gradient boosting
                "min_samples_split": [2, 5],  # Prevent too granular splits
                "min_samples_leaf": [1, 2],  # Minimum samples in leaves
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # GridSearchCV with Gradient Boosting (NO StandardScaler)
        from sklearn.ensemble import GradientBoostingRegressor

        self.grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=RANDOM_SEED, verbose=0),
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        # Log results
        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        # F = (R² / p) / ((1 - R²) / (n - p - 1))
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        metrics = {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

        return metrics

    def save_model(self, filepath: Path) -> None:
        # Save trained model to disk
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "GradientBoostingPredictor":
        # Load trained model from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_feature_importance(self) -> pd.Series:
        # Get feature importance rankings (sorted descending)
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.feature_importance_

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class ElasticNetPredictor:
    # Elastic Net Regressor with GridSearch and StandardScaler

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Initialize Elastic Net predictor
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.train_score_ = None
        self.cv_score_ = None
        self.overfitting_gap_ = None
        self.feature_names_ = None  # Store feature names for interpretability
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> "ElasticNetPredictor":
        # Fit Elastic Net with GridSearchCV and StandardScaler
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler

        if param_grid is None:
            # Pipeline requires 'model__' prefix for model parameters
            param_grid = {
                "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
                "model__l1_ratio": [
                    0.1,
                    0.3,
                    0.5,
                    0.7,
                    0.9,
                ],  # 0=Ridge, 1=Lasso, 0.5=balanced
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(
            f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
        )

        # Store feature names for interpretability
        self.feature_names_ = X_train.columns.tolist()

        # Pipeline prevents data leakage: scaler is refit in each CV fold
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=10000, random_state=RANDOM_SEED)),
            ]
        )

        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
        )

        self.grid_search.fit(X_train, y_train)

        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_

        self.logger.info(f"Best CV R^2 score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        self.train_score_ = cv_results.loc[best_idx, "mean_train_score"]
        self.cv_score_ = cv_results.loc[best_idx, "mean_test_score"]
        self.overfitting_gap_ = self.train_score_ - self.cv_score_

        self.logger.info(f"Mean train R^2: {self.train_score_:.4f}")
        self.logger.info(f"Mean CV R^2: {self.cv_score_:.4f}")
        self.logger.info(f"Train-CV gap: {self.overfitting_gap_:.4f}")

        if self.overfitting_gap_ > 0.1:
            self.logger.warning(
                f"[WARNING]  Potential overfitting detected (gap > 0.1)"
            )
        else:
            self.logger.info(f"[OK] No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Make predictions on new data
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        # Evaluate model on test set (R², Adjusted R², RMSE, MAE, F-statistic)
        y_pred = self.predict(X_test)

        n = len(y_test)
        p = X_test.shape[1]
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: penalizes for number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # F-statistic: overall model significance
        if r2 < 1.0 and p > 0:  # Avoid division by zero
            f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
            f_pvalue = float(1 - stats.f.cdf(f_stat, p, n - p - 1))
        else:
            f_stat = np.inf
            f_pvalue = 0.0

        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "n_samples": n,
            "n_features": p,
            # Overfitting metrics from GridSearchCV
            "train_score": self.train_score_ if self.train_score_ is not None else None,
            "cv_score": self.cv_score_ if self.cv_score_ is not None else None,
            "overfitting_gap": self.overfitting_gap_
            if self.overfitting_gap_ is not None
            else None,
            "has_overfitting": self.overfitting_gap_ > 0.1
            if self.overfitting_gap_ is not None
            else None,
        }

    def save_model(self, filepath: Path) -> None:
        """Save trained model (Pipeline with scaler) to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> "ElasticNetPredictor":
        # Load trained model (Pipeline with scaler) from disk
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_cv_results(self) -> pd.DataFrame:
        # Get detailed GridSearchCV results
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)

    def get_coefficients(self) -> pd.Series:
        # Get model coefficients with feature names (sorted by absolute value)
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if self.feature_names_ is None:
            raise ValueError(
                "Feature names not stored. This should not happen if fit() was called properly."
            )

        # Extract ElasticNet model from Pipeline
        elastic_net_model = self.model.named_steps["model"]

        # Create Series with feature names as index
        coefficients = pd.Series(elastic_net_model.coef_, index=self.feature_names_)

        # Sort by absolute value (most important features first)
        return coefficients.reindex(
            coefficients.abs().sort_values(ascending=False).index
        )


# ============================================================================
# PANEL DATA ECONOMETRIC MODELS
# ============================================================================


class PanelAnalyzer:
    # Panel data analysis with Dynamic Panel (Fixed Effects) estimation

    def __init__(
        self,
        data_path: Path,
        target: str,
        predictors: List[str],
        entity_col: str = "Country Name",
        time_col: str = "Year",
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize PanelAnalyzer
        self.data_path = Path(data_path)
        self.target = target
        self.predictors = predictors
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logger or self._create_logger()

        # Placeholders
        self.df: Optional[pd.DataFrame] = None
        self.df_panel: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.dynamic_results = None

    def _create_logger(self) -> logging.Logger:
        # Create default logger if none provided
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self) -> "PanelAnalyzer":
        # Load and validate panel dataset
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            raise IOError(f"Failed to read CSV file: {e}")

        if self.df.empty:
            raise ValueError("Loaded dataframe is empty")

        required_cols = [self.entity_col, self.time_col, self.target] + self.predictors
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.info(f"Loaded {len(self.df):,} observations")
        self.logger.info(f"Entities: {self.df[self.entity_col].nunique()}")
        self.logger.info(
            f"Period: {self.df[self.time_col].min():.0f} - {self.df[self.time_col].max():.0f}"
        )

        return self

    def prepare_panel_structure(self) -> "PanelAnalyzer":
        # Convert dataframe to panel structure with multi-index
        if self.df is None:
            raise RuntimeError("Must call load_data() first")

        if self.entity_col not in self.df.columns:
            raise ValueError(f"Entity column '{self.entity_col}' not found")
        if self.time_col not in self.df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found")

        try:
            self.df_panel = self.df.set_index(
                [self.entity_col, self.time_col]
            ).sort_index()
            self.logger.info("Panel structure created")
            self.logger.info(
                f"Entities: {self.df_panel.index.get_level_values(0).nunique()}"
            )
            self.logger.info(
                f"Time periods: {self.df_panel.index.get_level_values(1).nunique()}"
            )
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to create panel structure: {e}")

    def train_test_split(
        self,
        test_years: Optional[int] = None,
        train_end_year: Optional[int] = None,
        test_ratio: float = 0.2,
    ) -> "PanelAnalyzer":
        # Split panel data by time (temporal split for out-of-sample validation)
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        max_year = self.df_panel.index.get_level_values(1).max()
        min_year = self.df_panel.index.get_level_values(1).min()

        # Determine cutoff year
        if train_end_year is not None:
            # Use explicit train end year
            train_cutoff = train_end_year + 1
        elif test_years is not None:
            # Use test_years parameter (original behavior)
            train_cutoff = max_year - test_years + 1
        else:
            # Calculate based on test_ratio (default 80/20 split)
            total_years = max_year - min_year + 1
            test_years_calc = int(total_years * test_ratio)
            train_cutoff = max_year - test_years_calc + 1

        self.df_train = self.df_panel[
            self.df_panel.index.get_level_values(1) < train_cutoff
        ]
        self.df_test = self.df_panel[
            self.df_panel.index.get_level_values(1) >= train_cutoff
        ]

        train_years = self.df_train.index.get_level_values(1).unique()
        test_years_actual = self.df_test.index.get_level_values(1).unique()

        self.logger.info(f"Train/Test split: cutoff year = {train_cutoff}")
        self.logger.info(
            f"Training period: {train_years.min():.0f}-{train_years.max():.0f}"
        )
        self.logger.info(
            f"Test period: {test_years_actual.min():.0f}-{test_years_actual.max():.0f}"
        )
        self.logger.info(f"Training set: {len(self.df_train)} observations")
        self.logger.info(f"Test set: {len(self.df_test)} observations")

        return self

    def create_lagged_variable(self, variable: str, lags: int = 1) -> "PanelAnalyzer":
        # Create lagged variable for dynamic panel models
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        try:
            lag_name = f"{variable}_lag{lags}"
            self.df_panel[lag_name] = self.df_panel.groupby(level=0)[variable].shift(
                lags
            )

            if self.df_train is not None:
                self.df_train[lag_name] = self.df_train.groupby(level=0)[
                    variable
                ].shift(lags)
            if self.df_test is not None:
                self.df_test[lag_name] = self.df_test.groupby(level=0)[variable].shift(
                    lags
                )

            self.logger.info(f"Created lagged variable: {lag_name}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to create lagged variable: {e}")

    def fit_dynamic_panel(
        self, lags: int = 1, use_train_only: bool = False
    ) -> "PanelAnalyzer":
        # Fit dynamic panel model with lagged dependent variable
        # Create lagged variable and store lag info (Golden Rule B)
        lag_name = f"{self.target}_lag{lags}"
        self.dynamic_lags_ = lags
        self.dynamic_lag_name_ = lag_name

        if lag_name not in self.df_panel.columns:
            self.create_lagged_variable(self.target, lags)

        df = self.df_train if use_train_only else self.df_panel
        df_clean = df.dropna()

        self.logger.info("Fitting Dynamic Panel model...")
        self.logger.info(f"Observations after lagging: {len(df_clean)}")

        y_dynamic = df_clean[self.target]
        X_dynamic = df_clean[self.predictors + [lag_name]]

        try:
            dynamic_model = PanelOLS(
                y_dynamic, X_dynamic, entity_effects=True, time_effects=True
            )
            self.dynamic_results = dynamic_model.fit(
                cov_type="clustered", cluster_entity=True
            )

            self.logger.info(
                f"Dynamic Panel R^2 (within): {self.dynamic_results.rsquared_within:.4f}"
            )
            self.logger.info("Dynamic Panel model fitted successfully")

            return self
        except Exception as e:
            self.logger.error(f"Dynamic Panel model failed: {e}")
            raise

    def analyze_persistence(self) -> Dict[str, float]:
        # Analyze persistence of shocks using lagged coefficient
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        # Use stored lag name instead of hardcoding lag1
        lag_name = self.dynamic_lag_name_
        if lag_name not in self.dynamic_results.params.index:
            raise RuntimeError("Lagged variable not found in model")

        lag_coef = self.dynamic_results.params[lag_name]
        lag_pval = self.dynamic_results.pvalues[lag_name]

        significant = lag_pval < 0.05

        self.logger.info(f"Persistence coefficient (rho): {lag_coef:.4f}")
        self.logger.info(f"P-value: {lag_pval:.6f}")
        self.logger.info(f"Significant: {'YES' if significant else 'NO'}")

        result = {
            "coefficient": lag_coef,
            "p_value": lag_pval,
            "significant": significant,
        }

        if 0 < lag_coef < 1:
            half_life = -np.log(2) / np.log(lag_coef)
            interpretation = (
                "STRONG" if lag_coef > 0.7 else "MODERATE" if lag_coef > 0.4 else "WEAK"
            )
            result["half_life"] = half_life
            result["interpretation"] = interpretation

            self.logger.info(f"Half-life: {half_life:.2f} years")
            self.logger.info(f"Interpretation: {interpretation} persistence")

        return result

    def evaluate_on_test(self) -> Dict[str, float]:
        # Evaluate dynamic panel model on test set
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")
        if self.df_test is None:
            raise RuntimeError("Must call train_test_split() first")

        # Use stored lag name instead of hardcoding lag1
        lag_name = self.dynamic_lag_name_
        df_test_clean = self.df_test.dropna()

        if len(df_test_clean) == 0:
            raise ValueError("Test set is empty after removing NaN values")

        y_test = df_test_clean[self.target]
        X_test = df_test_clean[self.predictors + [lag_name]]

        # Get predictions (entity and time effects handled by linearmodels)
        try:
            y_pred = self.dynamic_results.predict(exog=X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            self.logger.info("=" * 50)
            self.logger.info("TEST SET EVALUATION")
            self.logger.info("=" * 50)
            self.logger.info(f"Test observations: {len(y_test)}")
            self.logger.info(f"RMSE: {rmse:.4f}")
            self.logger.info(f"R^2: {r2:.4f}")

            return {"n_test": len(y_test), "mse": mse, "rmse": rmse, "r2": r2}
        except Exception as e:
            self.logger.error(f"Test evaluation failed: {e}")
            raise

    def get_summary(self) -> pd.DataFrame:
        # Get summary of all fitted models
        summary_data = []

        if self.dynamic_results is not None:
            summary_data.append(
                {
                    "Model": "Dynamic Panel (FE)",
                    "R^2 (within)": self.dynamic_results.rsquared_within,
                    "R^2 (overall)": self.dynamic_results.rsquared_overall,
                    "N": self.dynamic_results.nobs,
                }
            )

        return pd.DataFrame(summary_data)

    # =========================================================================
    # ECONOMETRIC DIAGNOSTIC TESTS (POST-ESTIMATION)
    # =========================================================================

    def run_diagnostic_tests(self) -> Dict[str, Dict]:
        # Run econometric diagnostic tests (Hausman, autocorrelation, heteroscedasticity)
        if self.dynamic_results is None:
            raise RuntimeError(
                "Must fit dynamic panel model first. Call fit_dynamic_panel()"
            )

        self.logger.info("\n" + "=" * 70)
        self.logger.info("ECONOMETRIC DIAGNOSTIC TESTS")
        self.logger.info("=" * 70)

        diagnostics = {}

        # Test 1: Hausman Test (FE vs RE)
        try:
            diagnostics["hausman"] = self._hausman_test()
        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            diagnostics["hausman"] = {"error": str(e)}

        # Test 2: Autocorrelation Test
        try:
            diagnostics["autocorrelation"] = self._test_autocorrelation()
        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed: {e}")
            diagnostics["autocorrelation"] = {"error": str(e)}

        # Test 3: Heteroscedasticity Test
        try:
            diagnostics["heteroscedasticity"] = self._test_heteroscedasticity()
        except Exception as e:
            self.logger.warning(f"Heteroscedasticity test failed: {e}")
            diagnostics["heteroscedasticity"] = {"error": str(e)}

        self.logger.info("=" * 70)

        return diagnostics

    def _hausman_test(self) -> Dict[str, float]:
        # Hausman Test: Fixed Effects vs Random Effects specification
        self.logger.info("\n1. HAUSMAN TEST (Fixed Effects vs Random Effects)")
        self.logger.info("-" * 70)

        try:
            # Golden Rule A: Use exact sample from fitted model
            y = self.dynamic_results.model.dependent
            X = self.dynamic_results.model.exog

            # Refit FE/RE on same sample with unadjusted covariance for Hausman
            fe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
                cov_type="unadjusted"
            )
            re = RandomEffects(y, X).fit(cov_type="unadjusted")

            b_fe = fe.params
            b_re = re.params

            V_fe = fe.cov
            V_re = re.cov

            diff = b_fe - b_re
            V_diff = V_fe - V_re

            # Stable inversion (use pseudo-inverse if needed)
            try:
                inv_V = np.linalg.inv(V_diff.values)
            except np.linalg.LinAlgError:
                inv_V = np.linalg.pinv(V_diff.values)

            stat = float(diff.values.T @ inv_V @ diff.values)
            df = int(len(diff))
            pval = float(1 - stats.chi2.cdf(stat, df))

            conclusion = "Préférer FE (RE rejeté)" if pval < 0.05 else "RE acceptable"

            self.logger.info(f"Hausman chi^2 statistic: {stat:.4f}")
            self.logger.info(f"Degrees of freedom: {df}")
            self.logger.info(f"P-value: {pval:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")

            return {
                "test": "Hausman (unadjusted cov)",
                "statistic": stat,
                "df": df,
                "p_value": pval,
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            return {
                "test": "Hausman (unadjusted cov)",
                "error": str(e),
                "conclusion": "Using Fixed Effects (standard for panel data)",
            }

    def _test_autocorrelation(self) -> Dict[str, float]:
        # AR(1) test on panel residuals via auxiliary regression
        self.logger.info("\n2. AUTOCORRELATION TEST (AR1 Auxiliary Regression)")
        self.logger.info("-" * 70)

        try:
            import statsmodels.api as sm

            # Get residuals from fitted model
            resids = self.dynamic_results.resids.copy()
            df = pd.DataFrame({"e": resids})
            df["entity"] = df.index.get_level_values(0)
            df["time"] = df.index.get_level_values(1)

            # Lag residuals by entity
            df["e_lag"] = df.groupby("entity")["e"].shift(1)
            df = df.dropna(subset=["e", "e_lag"])

            if len(df) == 0:
                raise ValueError("Insufficient data after lagging residuals")

            # Auxiliary regression: e_it = rho * e_i,t-1 + u_it (no constant)
            aux = sm.OLS(df["e"].values, df[["e_lag"]].values).fit()

            rho = float(aux.params[0])
            pval = float(aux.pvalues[0])

            conclusion = (
                "Autocorrélation (AR1) détectée"
                if pval < 0.05
                else "Pas d'évidence d'AR1"
            )

            self.logger.info(f"Rho (AR1 coefficient): {rho:.4f}")
            self.logger.info(f"P-value: {pval:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")
            self.logger.info(
                f"Note: Model uses clustered SE to handle potential autocorrelation"
            )

            return {
                "test": "Auxiliary AR(1) on residuals",
                "rho": rho,
                "p_value": pval,
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Autocorrelation test failed: {e}")
            return {
                "test": "Auxiliary AR(1) on residuals",
                "error": str(e),
                "conclusion": "Test failed - using clustered SE as precaution",
            }

    def _test_heteroscedasticity(self) -> Dict[str, float]:
        # Breusch-Pagan Test: Heteroscedasticity in residuals
        self.logger.info("\n3. HETEROSCEDASTICITY TEST (Breusch-Pagan)")
        self.logger.info("-" * 70)

        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_breuschpagan

            # Golden Rule A: Use exact sample from fitted model
            # Extract raw numpy arrays from PanelData objects
            resid = self.dynamic_results.resids
            if hasattr(resid, "dataframe"):
                # PanelData object
                resid = resid.dataframe.values.ravel()
            elif hasattr(resid, "values"):
                resid = resid.values.ravel()
            else:
                resid = np.asarray(resid).ravel()

            X = self.dynamic_results.model.exog
            if hasattr(X, "dataframe"):
                # PanelData object
                X = X.dataframe.values
            elif hasattr(X, "values"):
                X = X.values
            else:
                X = np.asarray(X)

            # Ensure we have pure numpy arrays
            resid = np.asarray(resid, dtype=float).ravel()
            X = np.asarray(X, dtype=float)

            # BP expects a constant in exog
            X_bp = np.column_stack([np.ones(len(X)), X])

            lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, X_bp)

            conclusion = (
                "Hétéroscédasticité détectée"
                if lm_pvalue < 0.05
                else "Pas d'évidence d'hétéroscédasticité"
            )

            self.logger.info(f"LM statistic: {lm_stat:.4f}")
            self.logger.info(f"LM p-value: {lm_pvalue:.6f}")
            self.logger.info(f"F statistic: {f_stat:.4f}")
            self.logger.info(f"F p-value: {f_pvalue:.6f}")
            self.logger.info(f"Conclusion: {conclusion}")
            self.logger.info(
                f"Note: Model uses clustered SE to handle potential heteroscedasticity"
            )

            return {
                "test": "Breusch-Pagan",
                "lm_stat": float(lm_stat),
                "lm_pvalue": float(lm_pvalue),
                "f_stat": float(f_stat),
                "f_pvalue": float(f_pvalue),
                "conclusion": conclusion,
            }

        except Exception as e:
            self.logger.warning(f"Heteroscedasticity test failed: {e}")
            return {
                "test": "Breusch-Pagan",
                "error": str(e),
                "conclusion": "Test failed - using robust SE as precaution",
            }
