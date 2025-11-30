"""
Panel Data Analyzer - Object-Oriented Implementation
====================================================

A professional class-based implementation for dynamic panel data analysis.

Author: Data Science Team
Date: November 2025
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, RandomEffects
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant


class PanelAnalyzer:
    """
    A comprehensive panel data analyzer with Fixed Effects, Random Effects,
    and Dynamic Panel models.

    Attributes
    ----------
    data_path : Path
        Path to the panel data CSV file
    target : str
        Target variable name
    predictors : List[str]
        List of predictor variable names
    entity_col : str
        Entity dimension column name
    time_col : str
        Time dimension column name
    logger : logging.Logger
        Logger instance for tracking analysis
    df : pd.DataFrame
        Raw loaded dataframe
    df_panel : pd.DataFrame
        Panel-structured dataframe with MultiIndex
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
        Initialize the PanelAnalyzer.

        Parameters
        ----------
        data_path : Path
            Path to the panel data CSV file
        target : str
            Target variable name
        predictors : List[str]
            List of predictor variable names
        entity_col : str, default='Country Name'
            Entity dimension column name
        time_col : str, default='Year'
            Time dimension column name
        logger : logging.Logger, optional
            Logger instance (creates one if not provided)
        """
        self.data_path = data_path
        self.target = target
        self.predictors = predictors
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logger or self._create_logger()

        # Initialize containers for results
        self.df: Optional[pd.DataFrame] = None
        self.df_panel: Optional[pd.DataFrame] = None
        self.fe_results = None
        self.re_results = None
        self.dynamic_results = None

        self.logger.info("PanelAnalyzer initialized")

    def _create_logger(self) -> logging.Logger:
        """Create a default logger if none provided."""
        log_dir = self.data_path.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"panel_analyzer_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logger initialized. Log file: {log_file}")
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
        self.logger.info(f"Loading data from: {self.data_path}")

        if not self.data_path.exists():
            self.logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.debug(f"CSV file read successfully: {self.df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to read CSV file: {e}")
            raise IOError(f"Failed to read CSV file: {e}")

        if self.df.empty:
            self.logger.error("Loaded dataframe is empty")
            raise ValueError("Loaded dataframe is empty")

        required_cols = [self.entity_col, self.time_col, self.target] + self.predictors
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.info(f"✓ Data loaded: {len(self.df):,} rows, {self.df.shape[1]} columns")
        self.logger.info(f"  Entities: {self.df[self.entity_col].nunique()}")
        self.logger.info(f"  Time: {self.df[self.time_col].min():.0f} - {self.df[self.time_col].max():.0f}")

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
        RuntimeError
            If panel structure creation fails
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        self.logger.info(f"Creating panel structure: entity='{self.entity_col}', time='{self.time_col}'")

        try:
            self.df_panel = self.df.set_index([self.entity_col, self.time_col]).sort_index()
            self.logger.info(f"✓ Panel structure created")
            self.logger.info(f"  Entities: {self.df_panel.index.get_level_values(0).nunique()}")
            self.logger.info(f"  Time periods: {self.df_panel.index.get_level_values(1).nunique()}")
            return self
        except Exception as e:
            self.logger.error(f"Failed to create panel structure: {e}")
            raise RuntimeError(f"Failed to create panel structure: {e}")

    def fit_fixed_effects(self) -> 'PanelAnalyzer':
        """
        Fit Fixed Effects model with entity and time effects.

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        if self.df_panel is None:
            raise RuntimeError("Panel structure not prepared. Call prepare_panel_structure() first.")

        self.logger.info("Fitting Fixed Effects model")

        try:
            y = self.df_panel[self.target]
            X = self.df_panel[self.predictors]

            fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
            self.fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

            self.logger.info("✓ Fixed Effects model fitted")
            self.logger.info(f"  R² (within): {self.fe_results.rsquared_within:.4f}")
            self.logger.info(f"  R² (overall): {self.fe_results.rsquared_overall:.4f}")
            self.logger.info(f"  F-statistic: {self.fe_results.f_statistic.stat:.2f} (p={self.fe_results.f_statistic.pval:.6f})")

            return self
        except Exception as e:
            self.logger.error(f"Fixed Effects model failed: {e}", exc_info=True)
            raise

    def fit_random_effects(self) -> 'PanelAnalyzer':
        """
        Fit Random Effects model.

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        if self.df_panel is None:
            raise RuntimeError("Panel structure not prepared. Call prepare_panel_structure() first.")

        self.logger.info("Fitting Random Effects model")

        try:
            y = self.df_panel[self.target]
            X = self.df_panel[self.predictors]

            re_model = RandomEffects(y, X)
            self.re_results = re_model.fit(cov_type='clustered', cluster_entity=True)

            self.logger.info("✓ Random Effects model fitted")
            self.logger.info(f"  R² (overall): {self.re_results.rsquared_overall:.4f}")

            return self
        except Exception as e:
            self.logger.error(f"Random Effects model failed: {e}", exc_info=True)
            raise

    def hausman_test(self) -> Dict[str, float]:
        """
        Perform Hausman test to choose between FE and RE.

        Returns
        -------
        Dict[str, float]
            Test results with statistic, p-value, and decision

        Raises
        ------
        RuntimeError
            If FE or RE models not fitted
        """
        if self.fe_results is None or self.re_results is None:
            raise RuntimeError("Both FE and RE models must be fitted first.")

        self.logger.info("Performing Hausman test (FE vs RE)")

        try:
            b_fe = self.fe_results.params[self.predictors]
            b_re = self.re_results.params[self.predictors]
            V_fe = self.fe_results.cov[self.predictors].loc[self.predictors]
            V_re = self.re_results.cov[self.predictors].loc[self.predictors]

            diff = b_fe - b_re
            var_diff = V_fe - V_re

            hausman_stat = float(diff.T @ np.linalg.inv(var_diff) @ diff)
            df_test = len(self.predictors)
            p_value = 1 - stats.chi2.cdf(hausman_stat, df_test)

            decision = 'Use Fixed Effects' if p_value < 0.05 else 'Use Random Effects'

            self.logger.info(f"✓ Hausman test completed")
            self.logger.info(f"  Test statistic: {hausman_stat:.4f}")
            self.logger.info(f"  P-value: {p_value:.6f}")
            self.logger.info(f"  Decision: {decision}")

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
                'decision': 'Test failed - use FE by default'
            }

    def breusch_pagan_test(self) -> Dict[str, float]:
        """
        Breusch-Pagan test for heteroskedasticity.

        Returns
        -------
        Dict[str, float]
            Test results

        Raises
        ------
        RuntimeError
            If FE model not fitted
        """
        if self.fe_results is None:
            raise RuntimeError("FE model must be fitted first.")

        self.logger.info("Performing Breusch-Pagan test")

        try:
            X = self.df_panel[self.predictors]
            residuals = self.fe_results.resids

            X_with_const = add_constant(X, has_constant='add')
            lm_stat, lm_pval, fstat, f_pval = het_breuschpagan(residuals, X_with_const)

            decision = 'Heteroskedasticity detected' if lm_pval < 0.05 else 'Homoskedasticity'

            self.logger.info("✓ Breusch-Pagan test completed")
            self.logger.info(f"  LM statistic: {lm_stat:.4f}")
            self.logger.info(f"  P-value: {lm_pval:.6f}")
            self.logger.info(f"  Decision: {decision}")

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
        Wooldridge test for serial correlation.

        Returns
        -------
        Dict[str, float]
            Test results

        Raises
        ------
        RuntimeError
            If panel structure not prepared
        """
        if self.df_panel is None:
            raise RuntimeError("Panel structure not prepared.")

        self.logger.info("Performing Wooldridge test")

        try:
            from sklearn.linear_model import LinearRegression

            df_sorted = self.df_panel.sort_index()
            numeric_cols = [self.target] + self.predictors
            df_numeric = df_sorted[numeric_cols]

            df_diff = df_numeric.groupby(level=0).diff()
            df_diff = df_diff.dropna()

            if len(df_diff) < 30:
                raise ValueError("Insufficient observations for test")

            y_diff = df_diff[self.target]
            X_diff = df_diff[self.predictors]

            model = LinearRegression()
            model.fit(X_diff, y_diff)
            residuals = pd.Series(
                y_diff.values - model.predict(X_diff),
                index=y_diff.index
            )

            resid_lag = residuals.groupby(level=0).shift(1)
            valid_idx = resid_lag.notna() & residuals.notna()

            if valid_idx.sum() < 30:
                raise ValueError("Insufficient observations for test")

            correlation = np.corrcoef(residuals[valid_idx], resid_lag[valid_idx])[0, 1]
            n = valid_idx.sum()
            test_stat = correlation * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

            decision = 'Serial correlation detected' if p_value < 0.05 else 'No serial correlation'

            self.logger.info("✓ Wooldridge test completed")
            self.logger.info(f"  Test statistic: {test_stat:.4f}")
            self.logger.info(f"  P-value: {p_value:.6f}")
            self.logger.info(f"  Decision: {decision}")

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

    def fit_dynamic_panel(self, lags: int = 1) -> 'PanelAnalyzer':
        """
        Fit dynamic panel model with lagged dependent variable.

        Parameters
        ----------
        lags : int, default=1
            Number of lags for dependent variable

        Returns
        -------
        PanelAnalyzer
            Self for method chaining
        """
        if self.df_panel is None:
            raise RuntimeError("Panel structure not prepared.")

        self.logger.info(f"Fitting Dynamic Panel model with {lags} lag(s)")

        try:
            # Create lagged variable
            df_dynamic = self.df_panel.copy()
            lag_name = f'{self.target}_lag{lags}'
            df_dynamic[lag_name] = df_dynamic.groupby(level=0)[self.target].shift(lags)
            df_dynamic = df_dynamic.dropna()

            self.logger.info(f"✓ Lagged variable created: {lag_name}")
            self.logger.info(f"  Observations: {len(df_dynamic):,}")
            self.logger.info(f"  Lost: {len(self.df_panel) - len(df_dynamic):,}")

            y_dynamic = df_dynamic[self.target]
            X_dynamic = df_dynamic[self.predictors + [lag_name]]

            dynamic_fe = PanelOLS(y_dynamic, X_dynamic, entity_effects=True, time_effects=True)
            self.dynamic_results = dynamic_fe.fit(cov_type='clustered', cluster_entity=True)

            self.logger.info("✓ Dynamic Panel model fitted")
            self.logger.info(f"  R² (within): {self.dynamic_results.rsquared_within:.4f}")

            return self
        except Exception as e:
            self.logger.error(f"Dynamic Panel model failed: {e}", exc_info=True)
            raise

    def analyze_persistence(self, lags: int = 1) -> Dict[str, float]:
        """
        Analyze persistence from dynamic panel results.

        Parameters
        ----------
        lags : int, default=1
            Number of lags used in dynamic model

        Returns
        -------
        Dict[str, float]
            Persistence metrics

        Raises
        ------
        RuntimeError
            If dynamic model not fitted
        """
        if self.dynamic_results is None:
            raise RuntimeError("Dynamic model must be fitted first.")

        lag_var = f'{self.target}_lag{lags}'
        self.logger.info(f"Analyzing persistence coefficient for '{lag_var}'")

        try:
            lag_coef = self.dynamic_results.params[lag_var]
            lag_pval = self.dynamic_results.pvalues[lag_var]

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

                self.logger.info(f"✓ Persistence analysis completed")
                self.logger.info(f"  Coefficient (ρ): {lag_coef:.4f}")
                self.logger.info(f"  P-value: {lag_pval:.6f}")
                self.logger.info(f"  Half-life: {half_life:.2f} years")
                self.logger.info(f"  Strength: {strength}")
            else:
                self.logger.warning(f"Coefficient outside (0,1): {lag_coef:.4f}")

            return persistence_metrics
        except Exception as e:
            self.logger.error(f"Persistence analysis failed: {e}")
            raise

    def run_full_analysis(self) -> Dict[str, any]:
        """
        Run complete panel data analysis pipeline.

        Returns
        -------
        Dict[str, any]
            Dictionary containing all analysis results
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING FULL PANEL ANALYSIS")
        self.logger.info("=" * 100)

        # Load and prepare data
        self.load_data().prepare_panel_structure()

        # Fit models
        self.logger.info("-" * 100)
        self.fit_fixed_effects()

        self.logger.info("-" * 100)
        self.fit_random_effects()

        # Statistical tests
        self.logger.info("-" * 100)
        hausman = self.hausman_test()

        self.logger.info("-" * 100)
        bp_test = self.breusch_pagan_test()
        wt_test = self.wooldridge_test()

        # Dynamic panel
        self.logger.info("-" * 100)
        self.fit_dynamic_panel(lags=1)

        self.logger.info("-" * 100)
        persistence = self.analyze_persistence(lags=1)

        # Summary
        self.logger.info("=" * 100)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 100)
        self.logger.info(f"✓ Fixed Effects R² (within): {self.fe_results.rsquared_within:.4f}")
        self.logger.info(f"✓ Dynamic Panel R² (within): {self.dynamic_results.rsquared_within:.4f}")
        self.logger.info(f"✓ Model Selection: {hausman['decision']}")
        self.logger.info(f"✓ Heteroskedasticity: {bp_test['decision']}")
        self.logger.info(f"✓ Serial Correlation: {wt_test['decision']}")
        self.logger.info(f"✓ Persistence: ρ = {persistence['coefficient']:.4f} ({persistence['strength']})")
        self.logger.info("=" * 100)
        self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 100)

        return {
            'fe_results': self.fe_results,
            're_results': self.re_results,
            'dynamic_results': self.dynamic_results,
            'hausman': hausman,
            'breusch_pagan': bp_test,
            'wooldridge': wt_test,
            'persistence': persistence
        }


def main() -> int:
    """
    Main execution function demonstrating PanelAnalyzer usage.

    Returns
    -------
    int
        Exit code
    """
    try:
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

        # Create analyzer instance
        analyzer = PanelAnalyzer(
            data_path=data_file,
            target=target,
            predictors=predictors,
            entity_col='Country Name',
            time_col='Year'
        )

        # Run full analysis
        results = analyzer.run_full_analysis()

        return 0

    except Exception as e:
        logging.critical(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
