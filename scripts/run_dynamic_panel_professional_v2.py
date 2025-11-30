"""
Dynamic Panel Data Analysis - Professional Version with Logging
===============================================================

Improvements over v1:
- Professional logging system (replaces print statements)
- Log rotation and file management
- Structured error handling with detailed logging
- Performance tracking with timestamps

Author: Data Science Team
Date: November 2025
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, RandomEffects
from linearmodels.panel import compare
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings('ignore')


def setup_logging(log_dir: Path = None) -> logging.Logger:
    """
    Setup professional logging system.

    Parameters
    ----------
    log_dir : Path, optional
        Directory for log files. If None, creates logs/ in parent directory

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"panel_analysis_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def load_panel_data(data_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and validate panel dataset.

    Parameters
    ----------
    data_path : Path
        Path to the CSV file containing panel data
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Loaded and validated dataframe

    Raises
    ------
    FileNotFoundError
        If data file does not exist
    ValueError
        If data is empty or missing required columns
    """
    logger.info(f"Loading data from: {data_path}")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.debug(f"CSV file read successfully: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise IOError(f"Failed to read CSV file: {e}")

    if df.empty:
        logger.error("Loaded dataframe is empty")
        raise ValueError("Loaded dataframe is empty")

    required_cols = ['Country Name', 'Year', 'political_stability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"✓ Data loaded successfully: {len(df):,} rows, {df.shape[1]} columns")
    logger.info(f"  Countries: {df['Country Name'].nunique()}")
    logger.info(f"  Time period: {df['Year'].min():.0f} - {df['Year'].max():.0f}")

    return df


def prepare_panel_structure(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Convert dataframe to panel structure with multi-index.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    entity_col : str
        Column name for entity dimension
    time_col : str
        Column name for time dimension
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Panel-structured dataframe
    """
    logger.info(f"Creating panel structure with entity='{entity_col}', time='{time_col}'")

    if entity_col not in df.columns:
        logger.error(f"Entity column '{entity_col}' not found")
        raise ValueError(f"Entity column '{entity_col}' not found in dataframe")

    if time_col not in df.columns:
        logger.error(f"Time column '{time_col}' not found")
        raise ValueError(f"Time column '{time_col}' not found in dataframe")

    try:
        df_panel = df.set_index([entity_col, time_col]).sort_index()
        logger.info(f"✓ Panel structure created")
        logger.info(f"  Entities: {df_panel.index.get_level_values(0).nunique()}")
        logger.info(f"  Time periods: {df_panel.index.get_level_values(1).nunique()}")
        return df_panel
    except Exception as e:
        logger.error(f"Failed to create panel structure: {e}")
        raise RuntimeError(f"Failed to create panel structure: {e}")


def fit_fixed_effects(
    y: pd.Series,
    X: pd.DataFrame,
    logger: logging.Logger
):
    """
    Fit Fixed Effects model with entity and time effects.

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables
    logger : logging.Logger
        Logger instance

    Returns
    -------
    PanelOLS results
        Fitted model results
    """
    logger.info("Fitting Fixed Effects model with entity and time effects")

    try:
        fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

        logger.info("✓ Fixed Effects model fitted successfully")
        logger.info(f"  R² (within): {fe_results.rsquared_within:.4f}")
        logger.info(f"  R² (between): {fe_results.rsquared_between:.4f}")
        logger.info(f"  R² (overall): {fe_results.rsquared_overall:.4f}")
        logger.info(f"  F-statistic: {fe_results.f_statistic.stat:.2f} (p={fe_results.f_statistic.pval:.6f})")

        return fe_results
    except Exception as e:
        logger.error(f"Fixed Effects model fitting failed: {e}", exc_info=True)
        raise


def fit_random_effects(
    y: pd.Series,
    X: pd.DataFrame,
    logger: logging.Logger
):
    """
    Fit Random Effects model.

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables
    logger : logging.Logger
        Logger instance

    Returns
    -------
    RandomEffects results
        Fitted model results
    """
    logger.info("Fitting Random Effects model")

    try:
        re_model = RandomEffects(y, X)
        re_results = re_model.fit(cov_type='clustered', cluster_entity=True)

        logger.info("✓ Random Effects model fitted successfully")
        logger.info(f"  R² (overall): {re_results.rsquared_overall:.4f}")

        return re_results
    except Exception as e:
        logger.error(f"Random Effects model fitting failed: {e}", exc_info=True)
        raise


def hausman_test(
    fe_results,
    re_results,
    predictors: List[str],
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Perform Hausman test to choose between FE and RE.

    Parameters
    ----------
    fe_results : PanelOLS results
        Fixed effects results
    re_results : RandomEffects results
        Random effects results
    predictors : List[str]
        List of predictor names
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, float]
        Test results
    """
    logger.info("Performing Hausman test (FE vs RE)")

    try:
        b_fe = fe_results.params[predictors]
        b_re = re_results.params[predictors]
        V_fe = fe_results.cov[predictors].loc[predictors]
        V_re = re_results.cov[predictors].loc[predictors]

        diff = b_fe - b_re
        var_diff = V_fe - V_re

        hausman_stat = float(diff.T @ np.linalg.inv(var_diff) @ diff)
        df_test = len(predictors)
        p_value = 1 - stats.chi2.cdf(hausman_stat, df_test)

        decision = 'Use Fixed Effects' if p_value < 0.05 else 'Use Random Effects'

        logger.info(f"✓ Hausman test completed")
        logger.info(f"  Test statistic: {hausman_stat:.4f}")
        logger.info(f"  P-value: {p_value:.6f}")
        logger.info(f"  Decision: {decision}")

        return {
            'statistic': hausman_stat,
            'p_value': p_value,
            'df': df_test,
            'decision': decision
        }
    except Exception as e:
        logger.warning(f"Hausman test failed: {e}")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'df': len(predictors),
            'decision': 'Test failed - use FE by default'
        }


def create_lagged_variable(
    df_panel: pd.DataFrame,
    variable: str,
    lags: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Create lagged variable for dynamic panel.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Panel dataframe
    variable : str
        Variable to lag
    lags : int
        Number of lags
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DataFrame
        Dataframe with lagged variable
    """
    logger.info(f"Creating lag-{lags} variable for '{variable}'")

    try:
        df_result = df_panel.copy()
        lag_name = f'{variable}_lag{lags}'
        df_result[lag_name] = df_result.groupby(level=0)[variable].shift(lags)

        logger.info(f"✓ Lagged variable created: {lag_name}")
        return df_result
    except Exception as e:
        logger.error(f"Failed to create lagged variable: {e}")
        raise RuntimeError(f"Failed to create lagged variable: {e}")


def fit_dynamic_panel(
    y: pd.Series,
    X: pd.DataFrame,
    logger: logging.Logger
):
    """
    Fit dynamic panel model with lagged dependent variable.

    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables (including lag)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    PanelOLS results
        Fitted model results
    """
    logger.info("Fitting Dynamic Panel model")

    try:
        dynamic_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
        dynamic_results = dynamic_fe.fit(cov_type='clustered', cluster_entity=True)

        logger.info("✓ Dynamic Panel model fitted successfully")
        logger.info(f"  R² (within): {dynamic_results.rsquared_within:.4f}")
        logger.info(f"  Observations: {dynamic_results.nobs:,.0f}")

        return dynamic_results
    except Exception as e:
        logger.error(f"Dynamic Panel model fitting failed: {e}", exc_info=True)
        raise


def analyze_persistence(
    results,
    target: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Analyze persistence from dynamic panel results.

    Parameters
    ----------
    results : PanelOLS results
        Dynamic panel results
    target : str
        Target variable name
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, float]
        Persistence metrics
    """
    lag_var = f'{target}_lag1'
    logger.info(f"Analyzing persistence coefficient for '{lag_var}'")

    try:
        lag_coef = results.params[lag_var]
        lag_pval = results.pvalues[lag_var]

        persistence_metrics = {
            'coefficient': lag_coef,
            'p_value': lag_pval,
            'significant': lag_pval < 0.05
        }

        if 0 < lag_coef < 1:
            half_life = -np.log(2) / np.log(lag_coef)
            persistence_metrics['half_life'] = half_life

            if lag_coef > 0.7:
                strength = 'STRONG'
            elif lag_coef > 0.4:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'
            persistence_metrics['strength'] = strength

            logger.info(f"✓ Persistence analysis completed")
            logger.info(f"  Coefficient (ρ): {lag_coef:.4f}")
            logger.info(f"  P-value: {lag_pval:.6f}")
            logger.info(f"  Half-life: {half_life:.2f} years")
            logger.info(f"  Strength: {strength}")
        else:
            logger.warning(f"Persistence coefficient outside (0,1): {lag_coef:.4f}")

        return persistence_metrics
    except Exception as e:
        logger.error(f"Persistence analysis failed: {e}")
        raise


def wooldridge_test_serial_correlation(
    df_panel: pd.DataFrame,
    target: str,
    predictors: List[str],
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Wooldridge test for serial correlation in panel data.

    H0: No first-order autocorrelation

    Parameters
    ----------
    df_panel : pd.DataFrame
        Panel dataframe with multi-index
    target : str
        Dependent variable name
    predictors : List[str]
        Independent variable names
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, float]
        Test results with statistic and p-value
    """
    logger.info("Performing Wooldridge test for serial correlation")

    try:
        # First differences - select only numeric columns
        # Note: Differencing mixed-type DataFrames (numeric + string indices) causes
        # "unsupported operand type" errors. Extract numeric columns first.
        df_sorted = df_panel.sort_index()

        # Select all numeric columns
        numeric_cols = [target] + predictors
        df_numeric = df_sorted[numeric_cols]

        # Compute differences within each entity
        df_diff = df_numeric.groupby(level=0).diff()

        # Drop NaN from differencing
        df_diff = df_diff.dropna()

        if len(df_diff) < 30:
            logger.warning("Insufficient observations for Wooldridge test (<30)")
            raise ValueError("Insufficient observations for test")

        # Regression of differenced model
        y_diff = df_diff[target]
        X_diff = df_diff[predictors]

        # Simple regression to get residuals
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_diff, y_diff)
        residuals = pd.Series(
            y_diff.values - model.predict(X_diff),
            index=y_diff.index
        )

        # Test for autocorrelation in residuals
        resid_lag = residuals.groupby(level=0).shift(1)
        valid_idx = resid_lag.notna() & residuals.notna()

        if valid_idx.sum() < 30:
            logger.warning("Insufficient observations for Wooldridge test (<30)")
            raise ValueError("Insufficient observations for test")

        correlation = np.corrcoef(
            residuals[valid_idx],
            resid_lag[valid_idx]
        )[0, 1]

        n = valid_idx.sum()
        test_stat = correlation * np.sqrt(n)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

        decision = 'Serial correlation detected' if p_value < 0.05 else 'No serial correlation'

        logger.info("✓ Wooldridge test completed")
        logger.info(f"  Test statistic: {test_stat:.4f}")
        logger.info(f"  P-value: {p_value:.6f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Decision: {decision}")

        return {
            'statistic': test_stat,
            'p_value': p_value,
            'correlation': correlation,
            'decision': decision
        }
    except Exception as e:
        logger.warning(f"Wooldridge test failed: {e}")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'correlation': np.nan,
            'decision': 'Test failed'
        }


def breusch_pagan_test(
    model_results,
    X: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Breusch-Pagan test for heteroskedasticity.

    H0: Homoskedasticity (constant variance)

    Parameters
    ----------
    model_results : linearmodels results
        Fitted model results
    X : pd.DataFrame
        Independent variables
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, float]
        Test results
    """
    logger.info("Performing Breusch-Pagan test for heteroskedasticity")

    try:
        from statsmodels.tools import add_constant

        residuals = model_results.resids

        # Add constant to X for the test
        # Note: Breusch-Pagan requires at least 2 columns including a constant term
        # for auxiliary regression of squared residuals on regressors
        X_with_const = add_constant(X, has_constant='add')

        lm_stat, lm_pval, fstat, f_pval = het_breuschpagan(
            residuals,
            X_with_const
        )

        decision = 'Heteroskedasticity detected' if lm_pval < 0.05 else 'Homoskedasticity'

        logger.info("✓ Breusch-Pagan test completed")
        logger.info(f"  LM statistic: {lm_stat:.4f}")
        logger.info(f"  P-value: {lm_pval:.6f}")
        logger.info(f"  Decision: {decision}")

        return {
            'lm_statistic': lm_stat,
            'lm_p_value': lm_pval,
            'f_statistic': fstat,
            'f_p_value': f_pval,
            'decision': decision
        }
    except Exception as e:
        logger.warning(f"Breusch-Pagan test failed: {e}")
        return {
            'lm_statistic': np.nan,
            'lm_p_value': np.nan,
            'f_statistic': np.nan,
            'f_p_value': np.nan,
            'decision': 'Test failed'
        }


def main() -> int:
    """
    Main execution function.

    Returns
    -------
    int
        Exit code
    """
    # Setup logging
    logger = setup_logging()

    logger.info("=" * 100)
    logger.info("DYNAMIC PANEL DATA ANALYSIS - PROFESSIONAL VERSION V2")
    logger.info("=" * 100)

    # Configuration
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    data_file = data_dir / 'final_clean_data.csv'

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
        # Load data
        df = load_panel_data(data_file, logger)

        # Prepare panel
        df_panel = prepare_panel_structure(df, 'Country Name', 'Year', logger)

        y = df_panel[target]
        X = df_panel[predictors]

        # Fixed Effects
        logger.info("-" * 100)
        fe_results = fit_fixed_effects(y, X, logger)

        # Random Effects
        logger.info("-" * 100)
        re_results = fit_random_effects(y, X, logger)

        # Hausman test
        logger.info("-" * 100)
        hausman = hausman_test(fe_results, re_results, predictors, logger)

        # Diagnostic Tests
        logger.info("-" * 100)
        logger.info("DIAGNOSTIC TESTS")
        logger.info("-" * 100)

        # Breusch-Pagan Test
        bp_test = breusch_pagan_test(fe_results, X, logger)

        # Wooldridge Test
        wt_test = wooldridge_test_serial_correlation(df_panel, target, predictors, logger)

        # Dynamic Panel
        logger.info("-" * 100)
        df_dynamic = create_lagged_variable(df_panel, target, 1, logger)
        df_dynamic = df_dynamic.dropna()

        logger.info(f"Dynamic panel observations: {len(df_dynamic):,}")
        logger.info(f"Observations lost: {len(df_panel) - len(df_dynamic):,}")

        y_dynamic = df_dynamic[target]
        X_dynamic = df_dynamic[predictors + [f'{target}_lag1']]

        dynamic_results = fit_dynamic_panel(y_dynamic, X_dynamic, logger)

        # Persistence
        logger.info("-" * 100)
        persistence = analyze_persistence(dynamic_results, target, logger)

        # Summary
        logger.info("=" * 100)
        logger.info("SUMMARY")
        logger.info("=" * 100)
        logger.info(f"✓ Fixed Effects R² (within): {fe_results.rsquared_within:.4f}")
        logger.info(f"✓ Dynamic Panel R² (within): {dynamic_results.rsquared_within:.4f}")
        logger.info(f"✓ Model Selection: {hausman['decision']}")
        logger.info(f"✓ Heteroskedasticity: {bp_test['decision']}")
        logger.info(f"✓ Serial Correlation: {wt_test['decision']}")
        logger.info(f"✓ Persistence: ρ = {persistence['coefficient']:.4f} ({persistence['strength']})")
        logger.info("=" * 100)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)

        return 0

    except Exception as e:
        logger.critical(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
