#!/usr/bin/env python3
"""
Main Entry Point for Political Stability Panel Analysis
Supports multiple modes: dynamic_panel, predict, benchmark
"""

import argparse
import sys
from pathlib import Path

from src.models.panel_models import PanelAnalyzer
from src.utils.logging_config import setup_logging


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
        choices=['dynamic_panel', 'predict', 'benchmark'],
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


if __name__ == "__main__":
    sys.exit(main())
