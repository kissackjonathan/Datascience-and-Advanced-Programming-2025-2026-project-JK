"""
Optuna-enhanced model implementations for XGBoost and MLP.

This module provides fit_optuna() methods that use Bayesian optimization
instead of grid search for more efficient hyperparameter tuning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Import RANDOM_SEED from models.py
import sys
sys.path.insert(0, '.')
from src.models import RANDOM_SEED


def fit_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    cv: int = 5,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fit XGBoost with Optuna Bayesian optimization.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials (default: 100)
        cv: Cross-validation folds (default: 5)
        n_jobs: Parallel jobs for XGBoost (default: -1)
        logger: Logger instance

    Returns:
        Dict containing model, best_params, cv_score, train_score, etc.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Starting Optuna Bayesian optimization with {n_trials} trials and {cv}-fold CV")

    def objective(trial):
        # Optuna suggests hyperparameters with intelligent Bayesian sampling
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        model = XGBRegressor(
            **params,
            random_state=RANDOM_SEED,
            objective='reg:squarederror',
            tree_method='auto',
            n_jobs=n_jobs
        )

        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1
        )

        return scores.mean()

    # Create Optuna study with MedianPruner
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,  # XGBoost already uses multi-threading
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    cv_score = study.best_value

    logger.info(f"Best CV R^2 score: {cv_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Train final model
    model = XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        objective='reg:squarederror',
        tree_method='auto',
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)

    # Calculate overfitting metrics
    train_score = model.score(X_train, y_train)
    overfitting_gap = train_score - cv_score

    logger.info(f"Train R^2: {train_score:.4f}")
    logger.info(f"Train-CV gap: {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        logger.warning(f"[WARNING] Potential overfitting detected (gap > 0.1)")
    else:
        logger.info(f"[OK] No significant overfitting (gap < 0.1)")

    # Feature importance
    feature_importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        'model': model,
        'best_params': best_params,
        'cv_score': cv_score,
        'train_score': train_score,
        'overfitting_gap': overfitting_gap,
        'feature_importance': feature_importance,
        'optuna_study': study
    }


def fit_mlp_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 80,
    cv: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Fit MLP with Optuna Bayesian optimization + Early Stopping (THREE-LEVEL optimization).

    LEVEL 1: Optuna trial-level pruning (stops bad trials early)
    LEVEL 2: Cross-validation
    LEVEL 3: MLPRegressor early stopping (internal validation set)

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials (default: 80)
        cv: Cross-validation folds (default: 5)
        logger: Logger instance

    Returns:
        Dict containing model, best_params, cv_score, train_score, etc.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Starting THREE-LEVEL optimization: Optuna + CV + Early Stopping")
    logger.info(f"Trials: {n_trials}, CV folds: {cv}")

    def objective(trial):
        # LEVEL 1: Optuna suggests hyperparameters
        params = {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes',
                [(50,), (100,), (100, 50), (100, 100), (150,), (150, 100), (200,)]
            ),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }

        # Pipeline with StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                **params,
                random_state=RANDOM_SEED,
                early_stopping=True,        # LEVEL 3: Early stopping ✅
                validation_fraction=0.1,    # 10% for validation
                max_iter=1000,
                n_iter_no_change=20,
            ))
        ])

        # LEVEL 2: Cross-validation with pruning
        scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            score = pipeline.score(X_val, y_val)
            scores.append(score)

            # Optuna pruning: stop if performing poorly
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # Create study with MedianPruner
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1
        )
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    # Get best parameters
    best_params = study.best_params
    cv_score = study.best_value

    logger.info(f"Best CV R^2 score: {cv_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Train final model with best parameters
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(
            **best_params,
            random_state=RANDOM_SEED,
            early_stopping=True,        # Keep early stopping!
            validation_fraction=0.1,
            max_iter=1000,
            n_iter_no_change=20,
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Calculate overfitting metrics
    train_score = pipeline.score(X_train, y_train)
    overfitting_gap = train_score - cv_score

    logger.info(f"Train R^2: {train_score:.4f}")
    logger.info(f"Train-CV gap: {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        logger.warning(f"[WARNING] Potential overfitting detected (gap > 0.1)")
    else:
        logger.info(f"[OK] No significant overfitting (gap < 0.1)")

    return {
        'model': pipeline,
        'best_params': best_params,
        'cv_score': cv_score,
        'train_score': train_score,
        'overfitting_gap': overfitting_gap,
        'optuna_study': study
    }
