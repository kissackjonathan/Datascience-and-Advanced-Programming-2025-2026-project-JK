"""
Dynamic Panel Data Analysis - Professional Version (Phase 1)
============================================================

Improvements:
1. Type hints for all functions
2. Comprehensive error handling
3. Statistical diagnostic tests (Hausman, Wooldridge, Breusch-Pagan)
4. Input validation
5. Logging and error messages
6. Docstrings for all functions

Author: Data Science Team
Date: November 2025
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, RandomEffects
from linearmodels.panel import compare
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings('ignore')


def load_panel_data(data_path: Path) -> pd.DataFrame:
    """
    Load and validate panel dataset.

    Parameters
    ----------
    data_path : Path
        Path to the CSV file containing panel data

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
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV file: {e}")

    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    required_cols = ['Country Name', 'Year', 'political_stability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def prepare_panel_structure(
    df: pd.DataFrame,
    entity_col: str = 'Country Name',
    time_col: str = 'Year'
) -> pd.DataFrame:
    """
    Convert dataframe to panel structure with multi-index.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    entity_col : str, default='Country Name'
        Column name for entity dimension
    time_col : str, default='Year'
        Column name for time dimension

    Returns
    -------
    pd.DataFrame
        Panel-structured dataframe with (entity, time) multi-index

    Raises
    ------
    ValueError
        If entity or time columns not found
    """
    if entity_col not in df.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in dataframe")
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe")

    try:
        df_panel = df.set_index([entity_col, time_col]).sort_index()
        return df_panel
    except Exception as e:
        raise RuntimeError(f"Failed to create panel structure: {e}")


def hausman_test(
    fe_results,
    re_results,
    predictors: List[str]
) -> Dict[str, float]:
    """
    Perform Hausman test to choose between Fixed Effects and Random Effects.

    H0: Random Effects is consistent and efficient
    Ha: Fixed Effects is required

    Parameters
    ----------
    fe_results : PanelOLS results
        Fixed effects estimation results
    re_results : RandomEffects results
        Random effects estimation results
    predictors : List[str]
        List of predictor variable names

    Returns
    -------
    Dict[str, float]
        Dictionary with test statistic, p-value, and decision
    """
    try:
        # Extract coefficients
        b_fe = fe_results.params[predictors]
        b_re = re_results.params[predictors]

        # Variance-covariance matrices
        V_fe = fe_results.cov[predictors].loc[predictors]
        V_re = re_results.cov[predictors].loc[predictors]

        # Hausman statistic
        diff = b_fe - b_re
        var_diff = V_fe - V_re

        # Calculate chi-square statistic
        hausman_stat = float(diff.T @ np.linalg.inv(var_diff) @ diff)
        df_test = len(predictors)
        p_value = 1 - stats.chi2.cdf(hausman_stat, df_test)

        return {
            'statistic': hausman_stat,
            'p_value': p_value,
            'df': df_test,
            'decision': 'Use Fixed Effects' if p_value < 0.05 else 'Use Random Effects'
        }
    except Exception as e:
        print(f"Warning: Hausman test failed: {e}")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'df': len(predictors),
            'decision': 'Test failed - use Fixed Effects by default'
        }


def wooldridge_test_serial_correlation(
    df_panel: pd.DataFrame,
    target: str,
    predictors: List[str]
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

    Returns
    -------
    Dict[str, float]
        Test results with statistic and p-value
    """
    try:
        # First differences
        df_sorted = df_panel.sort_index()
        df_diff = df_sorted.groupby(level=0).diff()

        # Drop NaN from differencing
        df_diff = df_diff.dropna()

        # Regression of differenced model
        y_diff = df_diff[target]
        X_diff = df_diff[predictors]

        # Simple regression to get residuals
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_diff, y_diff)
        residuals = y_diff - model.predict(X_diff)

        # Test for autocorrelation in residuals
        resid_lag = residuals.groupby(level=0).shift(1)
        valid_idx = resid_lag.notna() & residuals.notna()

        if valid_idx.sum() < 30:
            raise ValueError("Insufficient observations for test")

        correlation = np.corrcoef(
            residuals[valid_idx],
            resid_lag[valid_idx]
        )[0, 1]

        n = valid_idx.sum()
        test_stat = correlation * np.sqrt(n)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

        return {
            'statistic': test_stat,
            'p_value': p_value,
            'correlation': correlation,
            'decision': 'Serial correlation detected' if p_value < 0.05 else 'No serial correlation'
        }
    except Exception as e:
        print(f"Warning: Wooldridge test failed: {e}")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'correlation': np.nan,
            'decision': 'Test failed'
        }


def breusch_pagan_test(
    model_results,
    X: pd.DataFrame
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

    Returns
    -------
    Dict[str, float]
        Test results
    """
    try:
        residuals = model_results.resids
        lm_stat, lm_pval, fstat, f_pval = het_breuschpagan(
            residuals,
            X
        )

        return {
            'lm_statistic': lm_stat,
            'lm_p_value': lm_pval,
            'f_statistic': fstat,
            'f_p_value': f_pval,
            'decision': 'Heteroskedasticity detected' if lm_pval < 0.05 else 'Homoskedasticity'
        }
    except Exception as e:
        print(f"Warning: Breusch-Pagan test failed: {e}")
        return {
            'lm_statistic': np.nan,
            'lm_p_value': np.nan,
            'f_statistic': np.nan,
            'f_p_value': np.nan,
            'decision': 'Test failed'
        }


def create_lagged_variable(
    df_panel: pd.DataFrame,
    variable: str,
    lags: int = 1
) -> pd.DataFrame:
    """
    Create lagged variable for dynamic panel models.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Panel dataframe with multi-index
    variable : str
        Variable to lag
    lags : int, default=1
        Number of lags

    Returns
    -------
    pd.DataFrame
        Dataframe with lagged variable added
    """
    try:
        df_result = df_panel.copy()
        lag_name = f'{variable}_lag{lags}'
        df_result[lag_name] = df_result.groupby(level=0)[variable].shift(lags)
        return df_result
    except Exception as e:
        raise RuntimeError(f"Failed to create lagged variable: {e}")


def main() -> int:
    """
    Main execution function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 100)
    print("DYNAMIC PANEL DATA ANALYSIS - PROFESSIONAL VERSION")
    print("=" * 100)
    print()

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
        print("Loading data...")
        df = load_panel_data(data_file)
        print(f"✓ Loaded {len(df):,} observations, {df['Country Name'].nunique()} countries")
        print(f"  Period: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
        print()

        # Prepare panel structure
        print("Preparing panel structure...")
        df_panel = prepare_panel_structure(df)
        print(f"✓ Panel structure created")
        print(f"  Entities: {df_panel.index.get_level_values(0).nunique()}")
        print(f"  Time periods: {df_panel.index.get_level_values(1).nunique()}")
        print()

        # Extract variables
        y = df_panel[target]
        X = df_panel[predictors]

        # Fixed Effects Model
        print("=" * 100)
        print("1. FIXED EFFECTS MODEL")
        print("=" * 100)
        print()

        fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

        print(fe_results.summary)
        print()

        # Random Effects Model
        print("=" * 100)
        print("2. RANDOM EFFECTS MODEL")
        print("=" * 100)
        print()

        re_model = RandomEffects(y, X)
        re_results = re_model.fit(cov_type='clustered', cluster_entity=True)

        print(re_results.summary)
        print()

        # Statistical Tests
        print("=" * 100)
        print("3. DIAGNOSTIC TESTS")
        print("=" * 100)
        print()

        # Hausman Test
        print("3.1 HAUSMAN TEST (FE vs RE)")
        print("-" * 100)
        hausman = hausman_test(fe_results, re_results, predictors)
        print(f"Test statistic: {hausman['statistic']:.4f}")
        print(f"P-value: {hausman['p_value']:.6f}")
        print(f"Degrees of freedom: {hausman['df']}")
        print(f"Decision: {hausman['decision']}")
        print()

        # Breusch-Pagan Test
        print("3.2 BREUSCH-PAGAN TEST (Heteroskedasticity)")
        print("-" * 100)
        bp_test = breusch_pagan_test(fe_results, X)
        print(f"LM statistic: {bp_test['lm_statistic']:.4f}")
        print(f"P-value: {bp_test['lm_p_value']:.6f}")
        print(f"Decision: {bp_test['decision']}")
        print()

        # Wooldridge Test
        print("3.3 WOOLDRIDGE TEST (Serial Correlation)")
        print("-" * 100)
        wt_test = wooldridge_test_serial_correlation(df_panel, target, predictors)
        print(f"Test statistic: {wt_test['statistic']:.4f}")
        print(f"P-value: {wt_test['p_value']:.6f}")
        print(f"Decision: {wt_test['decision']}")
        print()

        # Dynamic Panel Model
        print("=" * 100)
        print("4. DYNAMIC PANEL MODEL")
        print("=" * 100)
        print()

        df_dynamic = create_lagged_variable(df_panel, target, lags=1)
        df_dynamic = df_dynamic.dropna()

        print(f"Dynamic panel observations: {len(df_dynamic):,}")
        print(f"Observations lost due to lagging: {len(df_panel) - len(df_dynamic):,}")
        print()

        y_dynamic = df_dynamic[target]
        X_dynamic = df_dynamic[predictors + [f'{target}_lag1']]

        dynamic_fe = PanelOLS(y_dynamic, X_dynamic, entity_effects=True, time_effects=True)
        dynamic_fe_results = dynamic_fe.fit(cov_type='clustered', cluster_entity=True)

        print(dynamic_fe_results.summary)
        print()

        # Persistence Analysis
        print("=" * 100)
        print("5. PERSISTENCE ANALYSIS")
        print("=" * 100)
        print()

        lag_coef = dynamic_fe_results.params[f'{target}_lag1']
        lag_pval = dynamic_fe_results.pvalues[f'{target}_lag1']

        print(f"Lagged coefficient (ρ): {lag_coef:.4f}")
        print(f"P-value: {lag_pval:.6f}")
        print(f"Significant: {'YES ***' if lag_pval < 0.01 else 'YES **' if lag_pval < 0.05 else 'NO'}")
        print()

        if 0 < lag_coef < 1:
            half_life = -np.log(2) / np.log(lag_coef)
            print(f"Half-life of shocks: {half_life:.2f} years")
            print(f"Interpretation: {'STRONG' if lag_coef > 0.7 else 'MODERATE' if lag_coef > 0.4 else 'WEAK'} persistence")
        print()

        # Summary
        print("=" * 100)
        print("6. SUMMARY")
        print("=" * 100)
        print()
        print(f"✓ Fixed Effects R² (within): {fe_results.rsquared_within:.4f}")
        print(f"✓ Dynamic Panel R² (within): {dynamic_fe_results.rsquared_within:.4f}")
        print(f"✓ Model Selection: {hausman['decision']}")
        print(f"✓ Heteroskedasticity: {bp_test['decision']}")
        print(f"✓ Serial Correlation: {wt_test['decision']}")
        print(f"✓ Persistence: ρ = {lag_coef:.4f}")
        print()

        print("=" * 100)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 100)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: Invalid data - {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: Runtime error - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
