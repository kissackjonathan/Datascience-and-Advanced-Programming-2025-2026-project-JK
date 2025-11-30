#!/usr/bin/env python3
"""
Main Entry Point for Political Stability Analysis

Supports multiple modes:
- dynamic_panel: Panel data econometrics with fixed/random effects
- ml_random_forest: Random Forest with GridSearch optimization
- ml_xgboost: XGBoost with GridSearch optimization
- predict: Future predictions (not yet implemented)
- benchmark: Model comparison (not yet implemented)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.models.panel_models import PanelAnalyzer
from src.models.ml_models import RandomForestPredictor, XGBoostPredictor
from src.utils.logging_config import setup_logging
from src.utils.data_utils import get_train_test_split


def main() -> int:
    """
    Main execution function with command-line interface.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description='Political Stability Panel Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dynamic_panel
  python main.py --mode dynamic_panel --test_years 3
  python main.py --mode predict --data_path data/processed/final_clean_data.csv
        """
    )

    parser.add_argument(
        '--mode',
        choices=['dynamic_panel', 'predict', 'benchmark', 'ml_random_forest', 'ml_xgboost'],
        default='dynamic_panel',
        help='Analysis mode to run (default: dynamic_panel)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/final_clean_data.csv',
        help='Path to CSV data file (relative to project root)'
    )

    parser.add_argument(
        '--test_years',
        type=int,
        default=None,
        help='Number of years for test set (optional)'
    )

    parser.add_argument(
        '--train_end_year',
        type=int,
        default=2017,
        help='Last year to include in training set (default: 2017 for optimal performance)'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2,
        help='Test set ratio if neither test_years nor train_end_year specified (default: 0.2)'
    )

    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(level=log_level)

    logger.info("=" * 100)
    logger.info("POLITICAL STABILITY PANEL DATA ANALYSIS")
    logger.info("=" * 100)
    logger.info(f"Mode: {args.mode}")
    if args.train_end_year is not None:
        logger.info(f"Train end year: {args.train_end_year}")
    elif args.test_years is not None:
        logger.info(f"Test years: {args.test_years}")
    else:
        logger.info(f"Test ratio: {args.test_ratio}")

    # Robust path handling: resolve relative to main.py location
    project_root = Path(__file__).resolve().parent
    data_path = project_root / args.data_path

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error(f"Please check the path and try again")
        return 1

    logger.info(f"Data path: {data_path}")
    logger.info("")

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

    try:
        if args.mode == 'dynamic_panel':
            run_dynamic_panel_analysis(
                data_path=data_path,
                target=target,
                predictors=predictors,
                test_years=args.test_years,
                train_end_year=args.train_end_year,
                test_ratio=args.test_ratio,
                logger=logger
            )

        elif args.mode == 'predict':
            logger.info("Prediction mode not yet implemented")
            return 1

        elif args.mode == 'benchmark':
            logger.info("Benchmark mode not yet implemented")
            return 1

        elif args.mode == 'ml_random_forest':
            run_ml_random_forest_analysis(
                data_path=data_path,
                target=target,
                predictors=predictors,
                train_end_year=args.train_end_year,
                logger=logger
            )

        elif args.mode == 'ml_xgboost':
            run_ml_xgboost_analysis(
                data_path=data_path,
                target=target,
                predictors=predictors,
                train_end_year=args.train_end_year,
                logger=logger
            )

        logger.info("")
        logger.info("=" * 100)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_dynamic_panel_analysis(
    data_path: Path,
    target: str,
    predictors: list,
    test_years: int,
    train_end_year: int,
    test_ratio: float,
    logger
) -> None:
    """
    Run complete dynamic panel analysis with train/test split.

    Parameters
    ----------
    data_path : Path
        Path to data file
    target : str
        Dependent variable
    predictors : list
        Independent variables
    test_years : int, optional
        Number of years for test set
    train_end_year : int, optional
        Last year to include in training set
    test_ratio : float
        Test set ratio (default 0.2 for 80/20 split)
    logger : logging.Logger
        Logger instance
    """
    # Initialize analyzer
    analyzer = PanelAnalyzer(
        data_path=data_path,
        target=target,
        predictors=predictors,
        logger=logger
    )

    # Load and prepare data
    analyzer.load_data().prepare_panel_structure()

    # Train/test split
    analyzer.train_test_split(
        test_years=test_years,
        train_end_year=train_end_year,
        test_ratio=test_ratio
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("1. FIXED EFFECTS MODEL (on training data)")
    logger.info("=" * 100)
    logger.info("")

    # Fit Fixed Effects on training data
    analyzer.fit_fixed_effects(use_train_only=True)
    print(analyzer.fe_results.summary)
    print()

    logger.info("")
    logger.info("=" * 100)
    logger.info("2. RANDOM EFFECTS MODEL (on training data)")
    logger.info("=" * 100)
    logger.info("")

    # Fit Random Effects on training data
    analyzer.fit_random_effects(use_train_only=True)
    print(analyzer.re_results.summary)
    print()

    logger.info("")
    logger.info("=" * 100)
    logger.info("3. DIAGNOSTIC TESTS")
    logger.info("=" * 100)
    logger.info("")

    # Hausman test
    logger.info("3.1 HAUSMAN TEST (FE vs RE)")
    logger.info("-" * 100)
    hausman = analyzer.hausman_test()
    logger.info("")

    # Breusch-Pagan test
    logger.info("3.2 BREUSCH-PAGAN TEST (Heteroskedasticity)")
    logger.info("-" * 100)
    bp = analyzer.breusch_pagan_test()
    logger.info("")

    # Wooldridge test
    logger.info("3.3 WOOLDRIDGE TEST (Serial Correlation)")
    logger.info("-" * 100)
    wt = analyzer.wooldridge_test()
    logger.info("")

    logger.info("")
    logger.info("=" * 100)
    logger.info("4. DYNAMIC PANEL MODEL (on training data)")
    logger.info("=" * 100)
    logger.info("")

    # Fit dynamic panel on training data
    analyzer.fit_dynamic_panel(lags=1, use_train_only=True)
    print(analyzer.dynamic_results.summary)
    print()

    logger.info("")
    logger.info("=" * 100)
    logger.info("5. PERSISTENCE ANALYSIS")
    logger.info("=" * 100)
    logger.info("")

    persistence = analyzer.analyze_persistence()
    logger.info("")

    logger.info("")
    logger.info("=" * 100)
    logger.info("6. OUT-OF-SAMPLE EVALUATION (on test set)")
    logger.info("=" * 100)
    logger.info("")

    test_metrics = analyzer.evaluate_on_test()
    logger.info("")

    logger.info("")
    logger.info("=" * 100)
    logger.info("7. SUMMARY")
    logger.info("=" * 100)
    logger.info("")

    summary = analyzer.get_summary()
    print(summary.to_string(index=False))
    print()

    logger.info(f"Model Selection: {hausman['decision']}")
    logger.info(f"Heteroskedasticity: {bp['decision']}")
    logger.info(f"Serial Correlation: {wt['decision']}")
    logger.info(f"Persistence: ρ = {persistence['coefficient']:.4f}")
    if 'half_life' in persistence:
        logger.info(f"Half-life: {persistence['half_life']:.2f} years")
        logger.info(f"Interpretation: {persistence['interpretation']} persistence")
    logger.info(f"Test Set RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test Set R²: {test_metrics['r2']:.4f}")


def run_ml_random_forest_analysis(
    data_path: Path,
    target: str,
    predictors: list,
    train_end_year: int,
    logger
) -> None:
    """
    Run Random Forest analysis with GridSearch optimization.

    Why Random Forest:
    - Captures non-linear relationships
    - Robust to outliers
    - Provides feature importance

    Parameters
    ----------
    data_path : Path
        Path to data file
    target : str
        Dependent variable
    predictors : list
        Independent variables
    train_end_year : int
        Last year to include in training set (default: 2017)
    logger : logging.Logger
        Logger instance
    """
    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")

    # Train/test split (CONSISTENT with panel models)
    logger.info("")
    logger.info("=" * 100)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 100)

    df_train, df_test = get_train_test_split(df, train_end_year=train_end_year)

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
    logger.info("")
    logger.info("=" * 100)
    logger.info("RANDOM FOREST TRAINING WITH GRIDSEARCH")
    logger.info("=" * 100)

    rf_model = RandomForestPredictor(logger=logger)
    rf_model.fit(X_train, y_train, cv=5)

    # Evaluate on test set
    logger.info("")
    logger.info("=" * 100)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 100)

    test_metrics = rf_model.evaluate(X_test, y_test)

    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")

    # Feature importance
    logger.info("")
    logger.info("=" * 100)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 100)

    feature_importance = rf_model.get_feature_importance()
    for feature, importance in feature_importance.items():
        logger.info(f"{feature:30s}: {importance:.4f}")

    # Save model (to project root)
    project_root = Path(__file__).parent
    model_dir = project_root / 'trained_models'
    model_path = model_dir / 'random_forest_political_stability.pkl'
    rf_model.save_model(model_path)

    # Save results (to project root)
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


def run_ml_xgboost_analysis(
    data_path: Path,
    target: str,
    predictors: list,
    train_end_year: int,
    logger
) -> None:
    """
    Run XGBoost analysis with GridSearch optimization.

    Why XGBoost:
    - Gradient boosting (sequential learning)
    - Strong regularization (L1/L2)
    - Often better than Random Forest

    Parameters
    ----------
    data_path : Path
        Path to data file
    target : str
        Dependent variable
    predictors : list
        Independent variables
    train_end_year : int
        Last year to include in training set (default: 2017)
    logger : logging.Logger
        Logger instance
    """
    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")

    # Train/test split (CONSISTENT with other models)
    logger.info("")
    logger.info("=" * 100)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 100)

    df_train, df_test = get_train_test_split(df, train_end_year=train_end_year)

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
    logger.info("")
    logger.info("=" * 100)
    logger.info("XGBOOST TRAINING WITH GRIDSEARCH")
    logger.info("=" * 100)

    xgb_model = XGBoostPredictor(logger=logger)
    xgb_model.fit(X_train, y_train, cv=5)

    # Evaluate on test set
    logger.info("")
    logger.info("=" * 100)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 100)

    test_metrics = xgb_model.evaluate(X_test, y_test)

    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")

    # Feature importance
    logger.info("")
    logger.info("=" * 100)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 100)

    feature_importance = xgb_model.get_feature_importance()
    for feature, importance in feature_importance.items():
        logger.info(f"{feature:30s}: {importance:.4f}")

    # Save model (to project root)
    project_root = Path(__file__).parent
    model_dir = project_root / 'trained_models'
    model_path = model_dir / 'xgboost_political_stability.pkl'
    xgb_model.save_model(model_path)

    # Save results (to project root)
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / 'ml_xgboost_results.txt'

    with open(results_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("XGBOOST RESULTS - POLITICAL STABILITY PREDICTION\n")
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
        for param, value in xgb_model.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Best CV R²: {xgb_model.grid_search.best_score_:.4f}\n\n")

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


if __name__ == "__main__":
    sys.exit(main())
