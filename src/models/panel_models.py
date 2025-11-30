"""
Panel Data Analysis Models
OOP implementation of panel econometric models with train/test split
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant


class PanelAnalyzer:
    """
    Comprehensive panel data analysis with train/test split.

    Attributes
    ----------
    data_path : Path
        Path to panel dataset
    target : str
        Dependent variable name
    predictors : List[str]
        Independent variable names
    entity_col : str
        Entity identifier column (e.g., 'Country Name')
    time_col : str
        Time identifier column (e.g., 'Year')
    logger : logging.Logger
        Logger instance
    df : pd.DataFrame
        Raw dataframe
    df_panel : pd.DataFrame
        Panel-structured dataframe with MultiIndex
    df_train : pd.DataFrame
        Training set
    df_test : pd.DataFrame
        Test set
    fe_results : PanelOLS results
        Fixed Effects model results
    re_results : RandomEffects results
        Random Effects model results
    dynamic_results : PanelOLS results
        Dynamic panel model results
    """

    def __init__(
        self,
        data_path: Path,
        target: str,
        predictors: List[str],
        entity_col: str = 'Country Name',
        time_col: str = 'Year',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PanelAnalyzer.

        Parameters
        ----------
        data_path : Path
            Path to CSV file containing panel data
        target : str
            Dependent variable name
        predictors : List[str]
            Independent variable names
        entity_col : str, default='Country Name'
            Entity identifier column
        time_col : str, default='Year'
            Time identifier column
        logger : logging.Logger, optional
            Logger instance. Creates default if not provided
        """
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
        self.fe_results = None
        self.re_results = None
        self.dynamic_results = None

    def _create_logger(self) -> logging.Logger:
        """Create default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self) -> 'PanelAnalyzer':
        """
        Load and validate panel dataset.

        Returns
        -------
        PanelAnalyzer
            Self for method chaining

        Raises
        ------
        FileNotFoundError
            If data file does not exist
        ValueError
            If data is empty or missing required columns
        """
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

    def prepare_panel_structure(self) -> 'PanelAnalyzer':
        """
        Convert dataframe to panel structure with multi-index.

        Returns
        -------
        PanelAnalyzer
            Self for method chaining

        Raises
        ------
        ValueError
            If entity or time columns not found
        """
        if self.df is None:
            raise RuntimeError("Must call load_data() first")

        if self.entity_col not in self.df.columns:
            raise ValueError(f"Entity column '{self.entity_col}' not found")
        if self.time_col not in self.df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found")

        try:
            self.df_panel = self.df.set_index([self.entity_col, self.time_col]).sort_index()
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
        test_ratio: float = 0.2
    ) -> 'PanelAnalyzer':
        """
        Split panel data by time (temporal split for out-of-sample validation).

        Parameters
        ----------
        test_years : int, optional
            Number of most recent years to use as test set
        train_end_year : int, optional
            Last year to include in training set (test starts from train_end_year + 1)
            Takes precedence over test_years if specified
        test_ratio : float, default=0.2
            Ratio of data to use for testing (used if neither test_years nor train_end_year specified)

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
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
        self.logger.info(f"Training period: {train_years.min():.0f}-{train_years.max():.0f}")
        self.logger.info(f"Test period: {test_years_actual.min():.0f}-{test_years_actual.max():.0f}")
        self.logger.info(f"Training set: {len(self.df_train)} observations")
        self.logger.info(f"Test set: {len(self.df_test)} observations")

        return self

    def fit_fixed_effects(self, use_train_only: bool = False) -> 'PanelAnalyzer':
        """
        Fit Fixed Effects model with entity and time effects.

        Parameters
        ----------
        use_train_only : bool, default=False
            If True, fit on training set only. Otherwise fit on full data

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        df = self.df_train if use_train_only else self.df_panel

        if df is None:
            raise RuntimeError("Must prepare data first")

        self.logger.info("Fitting Fixed Effects model...")

        y = df[self.target]
        X = df[self.predictors]

        try:
            fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
            self.fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

            self.logger.info(f"Fixed Effects R² (within): {self.fe_results.rsquared_within:.4f}")
            self.logger.info("Fixed Effects model fitted successfully")

            return self
        except Exception as e:
            self.logger.error(f"Fixed Effects model failed: {e}")
            raise

    def fit_random_effects(self, use_train_only: bool = False) -> 'PanelAnalyzer':
        """
        Fit Random Effects model.

        Parameters
        ----------
        use_train_only : bool, default=False
            If True, fit on training set only. Otherwise fit on full data

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        df = self.df_train if use_train_only else self.df_panel

        if df is None:
            raise RuntimeError("Must prepare data first")

        self.logger.info("Fitting Random Effects model...")

        y = df[self.target]
        X = df[self.predictors]

        try:
            re_model = RandomEffects(y, X)
            self.re_results = re_model.fit(cov_type='clustered', cluster_entity=True)

            self.logger.info("Random Effects model fitted successfully")

            return self
        except Exception as e:
            self.logger.error(f"Random Effects model failed: {e}")
            raise

    def hausman_test(self) -> Dict[str, float]:
        """
        Perform Hausman test to choose between Fixed Effects and Random Effects.

        H0: Random Effects is consistent and efficient
        Ha: Fixed Effects is required

        Returns
        -------
        Dict[str, float]
            Dictionary with test statistic, p-value, and decision
        """
        if self.fe_results is None or self.re_results is None:
            raise RuntimeError("Must fit both FE and RE models first")

        try:
            # Extract coefficients
            b_fe = self.fe_results.params[self.predictors]
            b_re = self.re_results.params[self.predictors]

            # Variance-covariance matrices
            V_fe = self.fe_results.cov[self.predictors].loc[self.predictors]
            V_re = self.re_results.cov[self.predictors].loc[self.predictors]

            # Hausman statistic
            diff = b_fe - b_re
            var_diff = V_fe - V_re

            # Calculate chi-square statistic
            hausman_stat = float(diff.T @ np.linalg.inv(var_diff) @ diff)
            df_test = len(self.predictors)
            p_value = 1 - stats.chi2.cdf(hausman_stat, df_test)

            decision = 'Use Fixed Effects' if p_value < 0.05 else 'Use Random Effects'

            self.logger.info(f"Hausman test: χ² = {hausman_stat:.4f}, p = {p_value:.6f}")
            self.logger.info(f"Decision: {decision}")

            return {
                'statistic': hausman_stat,
                'p_value': p_value,
                'df': df_test,
                'decision': decision
            }
        except Exception as e:
            self.logger.warning(f"Hausman test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'df': len(self.predictors),
                'decision': 'Test failed - use Fixed Effects by default'
            }

    def breusch_pagan_test(self) -> Dict[str, float]:
        """
        Breusch-Pagan test for heteroskedasticity.

        H0: Homoskedasticity (constant variance)

        Returns
        -------
        Dict[str, float]
            Test results
        """
        if self.fe_results is None:
            raise RuntimeError("Must fit Fixed Effects model first")

        try:
            residuals = self.fe_results.resids
            X = self.df_panel[self.predictors] if self.df_train is None else self.df_train[self.predictors]

            # Note: Breusch-Pagan requires at least 2 columns including a constant term
            # for auxiliary regression of squared residuals on regressors
            X_with_const = add_constant(X, has_constant='add')

            lm_stat, lm_pval, fstat, f_pval = het_breuschpagan(residuals, X_with_const)

            decision = 'Heteroskedasticity detected' if lm_pval < 0.05 else 'Homoskedasticity'

            self.logger.info(f"Breusch-Pagan test: LM = {lm_stat:.4f}, p = {lm_pval:.6f}")
            self.logger.info(f"Decision: {decision}")

            return {
                'lm_statistic': lm_stat,
                'lm_p_value': lm_pval,
                'f_statistic': fstat,
                'f_p_value': f_pval,
                'decision': decision
            }
        except Exception as e:
            self.logger.warning(f"Breusch-Pagan test failed: {e}")
            return {
                'lm_statistic': np.nan,
                'lm_p_value': np.nan,
                'f_statistic': np.nan,
                'f_p_value': np.nan,
                'decision': 'Test failed'
            }

    def wooldridge_test(self) -> Dict[str, float]:
        """
        Wooldridge test for serial correlation in panel data.

        H0: No first-order autocorrelation

        Returns
        -------
        Dict[str, float]
            Test results with statistic and p-value
        """
        df = self.df_panel if self.df_train is None else self.df_train

        if df is None:
            raise RuntimeError("Must prepare panel structure first")

        try:
            # Note: Differencing mixed-type DataFrames (numeric + string indices) causes
            # "unsupported operand type" errors. Extract numeric columns first.
            df_sorted = df.sort_index()
            numeric_cols = [self.target] + self.predictors
            df_numeric = df_sorted[numeric_cols]
            df_diff = df_numeric.groupby(level=0).diff()

            # Drop NaN from differencing
            df_diff = df_diff.dropna()

            # Regression of differenced model
            y_diff = df_diff[self.target]
            X_diff = df_diff[self.predictors]

            # Simple regression to get residuals
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
                raise ValueError("Insufficient observations for test")

            correlation = np.corrcoef(
                residuals[valid_idx],
                resid_lag[valid_idx]
            )[0, 1]

            n = valid_idx.sum()
            test_stat = correlation * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

            decision = 'Serial correlation detected' if p_value < 0.05 else 'No serial correlation'

            self.logger.info(f"Wooldridge test: stat = {test_stat:.4f}, p = {p_value:.6f}")
            self.logger.info(f"Decision: {decision}")

            return {
                'statistic': test_stat,
                'p_value': p_value,
                'correlation': correlation,
                'decision': decision
            }
        except Exception as e:
            self.logger.warning(f"Wooldridge test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'correlation': np.nan,
                'decision': 'Test failed'
            }

    def create_lagged_variable(
        self,
        variable: str,
        lags: int = 1
    ) -> 'PanelAnalyzer':
        """
        Create lagged variable for dynamic panel models.

        Parameters
        ----------
        variable : str
            Variable to lag
        lags : int, default=1
            Number of lags

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        if self.df_panel is None:
            raise RuntimeError("Must call prepare_panel_structure() first")

        try:
            lag_name = f'{variable}_lag{lags}'
            self.df_panel[lag_name] = self.df_panel.groupby(level=0)[variable].shift(lags)

            if self.df_train is not None:
                self.df_train[lag_name] = self.df_train.groupby(level=0)[variable].shift(lags)
            if self.df_test is not None:
                self.df_test[lag_name] = self.df_test.groupby(level=0)[variable].shift(lags)

            self.logger.info(f"Created lagged variable: {lag_name}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to create lagged variable: {e}")

    def fit_dynamic_panel(
        self,
        lags: int = 1,
        use_train_only: bool = False
    ) -> 'PanelAnalyzer':
        """
        Fit dynamic panel model with lagged dependent variable.

        Parameters
        ----------
        lags : int, default=1
            Number of lags for dependent variable
        use_train_only : bool, default=False
            If True, fit on training set only

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        # Create lagged variable
        lag_name = f'{self.target}_lag{lags}'
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
                y_dynamic,
                X_dynamic,
                entity_effects=True,
                time_effects=True
            )
            self.dynamic_results = dynamic_model.fit(
                cov_type='clustered',
                cluster_entity=True
            )

            self.logger.info(
                f"Dynamic Panel R² (within): {self.dynamic_results.rsquared_within:.4f}"
            )
            self.logger.info("Dynamic Panel model fitted successfully")

            return self
        except Exception as e:
            self.logger.error(f"Dynamic Panel model failed: {e}")
            raise

    def analyze_persistence(self) -> Dict[str, float]:
        """
        Analyze persistence of shocks using lagged coefficient.

        Returns
        -------
        Dict[str, float]
            Persistence analysis results
        """
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")

        lag_name = f'{self.target}_lag1'
        if lag_name not in self.dynamic_results.params.index:
            raise RuntimeError("Lagged variable not found in model")

        lag_coef = self.dynamic_results.params[lag_name]
        lag_pval = self.dynamic_results.pvalues[lag_name]

        significant = lag_pval < 0.05

        self.logger.info(f"Persistence coefficient (ρ): {lag_coef:.4f}")
        self.logger.info(f"P-value: {lag_pval:.6f}")
        self.logger.info(f"Significant: {'YES' if significant else 'NO'}")

        result = {
            'coefficient': lag_coef,
            'p_value': lag_pval,
            'significant': significant
        }

        if 0 < lag_coef < 1:
            half_life = -np.log(2) / np.log(lag_coef)
            interpretation = (
                'STRONG' if lag_coef > 0.7
                else 'MODERATE' if lag_coef > 0.4
                else 'WEAK'
            )
            result['half_life'] = half_life
            result['interpretation'] = interpretation

            self.logger.info(f"Half-life: {half_life:.2f} years")
            self.logger.info(f"Interpretation: {interpretation} persistence")

        return result

    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Evaluate dynamic panel model on test set.

        Returns
        -------
        Dict[str, float]
            Test set performance metrics
        """
        if self.dynamic_results is None:
            raise RuntimeError("Must fit dynamic panel model first")
        if self.df_test is None:
            raise RuntimeError("Must call train_test_split() first")

        lag_name = f'{self.target}_lag1'
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
            self.logger.info(f"R²: {r2:.4f}")

            return {
                'n_test': len(y_test),
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        except Exception as e:
            self.logger.error(f"Test evaluation failed: {e}")
            raise

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all fitted models.

        Returns
        -------
        pd.DataFrame
            Summary table with model statistics
        """
        summary_data = []

        if self.fe_results is not None:
            summary_data.append({
                'Model': 'Fixed Effects',
                'R² (within)': self.fe_results.rsquared_within,
                'N': self.fe_results.nobs
            })

        if self.re_results is not None:
            summary_data.append({
                'Model': 'Random Effects',
                'R² (overall)': self.re_results.rsquared_overall,
                'N': self.re_results.nobs
            })

        if self.dynamic_results is not None:
            summary_data.append({
                'Model': 'Dynamic Panel',
                'R² (within)': self.dynamic_results.rsquared_within,
                'N': self.dynamic_results.nobs
            })

        return pd.DataFrame(summary_data)
