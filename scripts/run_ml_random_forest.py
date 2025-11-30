#!/usr/bin/env python3
"""
Standalone script for Random Forest training and evaluation.

Usage:
    python scripts/run_ml_random_forest.py

Why this script:
- Allows experimentation without modifying main.py
- Can be run independently for quick iterations
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.ml_models import RandomForestPredictor
from src.utils.data_utils import get_train_test_split
from src.utils.logging_config import setup_logging


def main():
    """
    Train and evaluate Random Forest model.

    Train/Test split:
    - Train: 1996-2017 (22 years)
    - Test: 2018-2023 (6 years)

    Why 2017 cutoff:
    - Consistent with panel models (test R² = 0.6736)
    - Better performance than 2016 cutoff
    - See notebooks/05_train_test_split_comparison.ipynb
    """
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 100)
    logger.info("RANDOM FOREST TRAINING - POLITICAL STABILITY PREDICTION")
    logger.info("=" * 100)

    # Load data
    data_path = project_root / 'data' / 'processed' / 'final_clean_data.csv'
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")

    # Configuration
    target = 'political_stability'
    predictors = [
        'gdp_per_capita',
        'gdp_growth',
        'unemployment_ilo',
        'inflation_cpi',
        'trade_gdp_pct',
        'rule_of_law',
        'government_effectiveness',
        'hdi'
    ]

    # Train/test split (CONSISTENT with panel models)
    logger.info("\n" + "=" * 100)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 100)

    df_train, df_test = get_train_test_split(df, train_end_year=2017)

    logger.info(f"Train period: {df_train['Year'].min()}-{df_train['Year'].max()}")
    logger.info(f"Test period: {df_test['Year'].min()}-{df_test['Year'].max()}")
    logger.info(f"Train size: {len(df_train)} observations")
    logger.info(f"Test size: {len(df_test)} observations")

    # Prepare features and target
    X_train = df_train[predictors]
    y_train = df_train[target]
    X_test = df_test[predictors]
    y_test = df_test[target]

    # Drop missing values
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    logger.info(f"Train size (after dropping NaN): {len(X_train)}")
    logger.info(f"Test size (after dropping NaN): {len(X_test)}")

    # Initialize and train model
    logger.info("\n" + "=" * 100)
    logger.info("RANDOM FOREST TRAINING WITH GRIDSEARCH")
    logger.info("=" * 100)

    rf_model = RandomForestPredictor(logger=logger)

    # Custom parameter grid (optional - can modify for experiments)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_model.fit(X_train, y_train, param_grid=param_grid, cv=5)

    # Evaluate on test set
    logger.info("\n" + "=" * 100)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 100)

    test_metrics = rf_model.evaluate(X_test, y_test)

    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")

    # Feature importance
    logger.info("\n" + "=" * 100)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 100)

    feature_importance = rf_model.get_feature_importance()
    for feature, importance in feature_importance.items():
        logger.info(f"{feature:30s}: {importance:.4f}")

    # Save model
    model_dir = project_root / 'trained_models'
    model_path = model_dir / 'random_forest_political_stability.pkl'
    rf_model.save_model(model_path)

    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / 'ml_random_forest_results.txt'

    with open(results_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("RANDOM FOREST RESULTS - POLITICAL STABILITY PREDICTION\n")
        f.write("=" * 100 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Target: {target}\n")
        f.write(f"Predictors: {', '.join(predictors)}\n")
        f.write(f"Train period: {df_train['Year'].min()}-{df_train['Year'].max()}\n")
        f.write(f"Test period: {df_test['Year'].min()}-{df_test['Year'].max()}\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 100 + "\n")
        for param, value in rf_model.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Best CV R²: {rf_model.grid_search.best_score_:.4f}\n\n")

        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 100 + "\n")
        f.write(f"Test R²: {test_metrics['r2']:.4f}\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.4f}\n\n")

        f.write("FEATURE IMPORTANCE\n")
        f.write("-" * 100 + "\n")
        for feature, importance in feature_importance.items():
            f.write(f"{feature:30s}: {importance:.4f}\n")

    logger.info(f"\n✅ Results saved to: {results_file}")

    logger.info("\n" + "=" * 100)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
