"""
Machine Learning models for political stability prediction.

Why ML models complement panel analysis:
- Panel models: capture structural relationships and causality
- ML models: optimize predictive accuracy with non-linear patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from typing import Dict, Optional
import logging

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class RandomForestPredictor:
    """
    Random Forest Regressor with GridSearch hyperparameter optimization.

    Why Random Forest:
    - Handles non-linear relationships naturally
    - Robust to outliers and missing values
    - Provides feature importance rankings
    - Less prone to overfitting than single decision trees

    Why NO StandardScaler:
    - Tree-based models are invariant to monotonic transformations
    - Scaling doesn't affect splits or predictions
    - Scaling doesn't prevent overfitting for trees
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize Random Forest predictor."""
        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.feature_importance_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1
    ) -> 'RandomForestPredictor':
        """
        Fit Random Forest with GridSearchCV for hyperparameter optimization.

        Why GridSearchCV only (no additional cross_val_score):
        - GridSearchCV already does k-fold CV internally
        - grid_search.best_score_ gives CV performance
        - Running cross_val_score again is redundant

        Overfitting prevention strategy:
        1. max_depth: Limit tree depth to prevent memorization
        2. min_samples_split: Require minimum samples before splitting
        3. min_samples_leaf: Require minimum samples in leaf nodes
        4. max_features: Limit features per split (adds randomness)
        5. K-fold CV: Validate on multiple train/val splits

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        param_grid : dict, optional
            Hyperparameter grid for GridSearch
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs

        Returns
        -------
        self : RandomForestPredictor
            Fitted model
        """
        if param_grid is None:
            # Default grid with overfitting prevention
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],  # Prevent too deep trees
                'min_samples_split': [2, 5, 10],  # Require samples before split
                'min_samples_leaf': [1, 2, 4],    # Require samples in leaves
                'max_features': ['sqrt', 'log2']   # Limit features per split
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        # GridSearchCV with k-fold CV (NO StandardScaler, NO redundant cross_val_score)
        self.grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True  # To monitor overfitting
        )

        self.grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        # Log results
        self.logger.info(f"Best CV R² score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting by comparing train vs CV scores
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        train_score = cv_results.loc[best_idx, 'mean_train_score']
        val_score = cv_results.loc[best_idx, 'mean_test_score']
        overfitting_gap = train_score - val_score

        self.logger.info(f"Mean train R²: {train_score:.4f}")
        self.logger.info(f"Mean CV R²: {val_score:.4f}")
        self.logger.info(f"Train-CV gap: {overfitting_gap:.4f}")

        if overfitting_gap > 0.1:
            self.logger.warning(f"⚠️  Potential overfitting detected (gap > 0.1)")
        else:
            self.logger.info(f"✅ No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target

        Returns
        -------
        metrics : dict
            Dictionary with R², RMSE, MAE
        """
        y_pred = self.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'n_samples': len(y_test)
        }

        return metrics

    def save_model(self, filepath: Path) -> None:
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save model (e.g., 'trained_models/random_forest.pkl')
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> 'RandomForestPredictor':
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : Path
            Path to saved model

        Returns
        -------
        self : RandomForestPredictor
            Model with loaded weights
        """
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance rankings.

        Returns
        -------
        importance : pd.Series
            Feature importances sorted descending
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.feature_importance_

    def get_cv_results(self) -> pd.DataFrame:
        """
        Get detailed GridSearchCV results.

        Returns
        -------
        results : pd.DataFrame
            CV results with all parameter combinations
        """
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


class XGBoostPredictor:
    """
    XGBoost Regressor with GridSearch hyperparameter optimization.

    Why XGBoost:
    - Gradient boosting: learns sequentially from previous errors
    - Regularization: L1 (alpha) and L2 (lambda) prevent overfitting
    - Handles missing values natively
    - Generally better performance than Random Forest

    Why NO StandardScaler:
    - Tree-based model, scale-invariant like Random Forest
    - Scaling doesn't affect splits or predictions
    - Scaling doesn't prevent overfitting for trees

    Why NO early_stopping in GridSearchCV:
    - early_stopping needs eval_set, incompatible with GridSearchCV
    - Regularization + learning_rate control overfitting instead
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize XGBoost predictor."""
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.model = None
        self.grid_search = None
        self.best_params_ = None
        self.feature_importance_ = None
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1
    ) -> 'XGBoostPredictor':
        """
        Fit XGBoost with GridSearchCV for hyperparameter optimization.

        Why this approach (NO early_stopping):
        - early_stopping needs eval_set for each CV fold
        - GridSearchCV doesn't easily support different eval_sets per fold
        - Solution: Use regularization + small learning_rate instead

        Overfitting prevention strategy:
        1. learning_rate: Small values (0.03-0.1) prevent overfitting
        2. max_depth: Limit tree complexity (3-7)
        3. subsample: Use only fraction of data (0.8-1.0)
        4. colsample_bytree: Use only fraction of features (0.8-1.0)
        5. reg_alpha/lambda: L1/L2 regularization
        6. K-fold CV: Validate on multiple splits

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        param_grid : dict, optional
            Hyperparameter grid for GridSearch
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs

        Returns
        -------
        self : XGBoostPredictor
            Fitted model
        """
        if param_grid is None:
            # Default grid with reasonable overfitting prevention
            # 192 combinations → ~2-3 minutes execution time
            param_grid = {
                'n_estimators': [100, 200],              # Number of boosting rounds
                'max_depth': [3, 5, 7],                  # Tree depth (prevent overfitting)
                'learning_rate': [0.03, 0.1],            # Step size (small = less overfitting)
                'subsample': [0.8, 1.0],                 # Row sampling (like bagging)
                'colsample_bytree': [0.8, 1.0],          # Column sampling (like max_features)
                'reg_alpha': [0, 0.1],                   # L1 regularization
                'reg_lambda': [1, 2]                     # L2 regularization
            }

        self.logger.info(f"Starting GridSearchCV with {cv}-fold CV")
        self.logger.info(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        # GridSearchCV with XGBoost (NO early_stopping)
        self.grid_search = GridSearchCV(
            estimator=XGBRegressor(
                random_state=42,
                objective='reg:squarederror',
                tree_method='auto'
            ),
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )

        self.grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        # Log results
        self.logger.info(f"Best CV R² score: {self.grid_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params_}")

        # Check for overfitting by comparing train vs CV scores
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        best_idx = self.grid_search.best_index_
        train_score = cv_results.loc[best_idx, 'mean_train_score']
        val_score = cv_results.loc[best_idx, 'mean_test_score']
        overfitting_gap = train_score - val_score

        self.logger.info(f"Mean train R²: {train_score:.4f}")
        self.logger.info(f"Mean CV R²: {val_score:.4f}")
        self.logger.info(f"Train-CV gap: {overfitting_gap:.4f}")

        if overfitting_gap > 0.1:
            self.logger.warning(f"⚠️  Potential overfitting detected (gap > 0.1)")
        else:
            self.logger.info(f"✅ No significant overfitting (gap < 0.1)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target

        Returns
        -------
        metrics : dict
            Dictionary with R², RMSE, MAE
        """
        y_pred = self.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'n_samples': len(y_test)
        }

        return metrics

    def save_model(self, filepath: Path) -> None:
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save model (e.g., 'trained_models/xgboost.pkl')
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path) -> 'XGBoostPredictor':
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : Path
            Path to saved model

        Returns
        -------
        self : XGBoostPredictor
            Model with loaded weights
        """
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        return self

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance rankings.

        Returns
        -------
        importance : pd.Series
            Feature importances sorted descending
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.feature_importance_

    def get_cv_results(self) -> pd.DataFrame:
        """
        Get detailed GridSearchCV results.

        Returns
        -------
        results : pd.DataFrame
            CV results with all parameter combinations
        """
        if self.grid_search is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return pd.DataFrame(self.grid_search.cv_results_)
