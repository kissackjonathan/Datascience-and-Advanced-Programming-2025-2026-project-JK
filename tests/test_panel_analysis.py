"""
Unit tests for Dynamic Panel Analysis
======================================

Tests cover:
- Data loading and validation
- Panel structure preparation
- Model fitting (FE, RE, Dynamic)
- Statistical tests (Hausman)
- Lagged variable creation
- Persistence analysis

Run with: pytest tests/test_panel_analysis.py -v
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_dynamic_panel_professional_v2 import (
    analyze_persistence,
    create_lagged_variable,
    fit_dynamic_panel,
    fit_fixed_effects,
    fit_random_effects,
    hausman_test,
    load_panel_data,
    prepare_panel_structure,
    setup_logging,
)


@pytest.fixture
def sample_panel_data():
    """
    Create sample panel dataset for testing.

    Returns
    -------
    pd.DataFrame
        Sample panel data with 10 countries, 10 years (larger for RE model)
    """
    np.random.seed(42)

    countries = [f'Country{chr(65+i)}' for i in range(10)]  # CountryA to CountryJ
    years = list(range(2014, 2024))  # 10 years

    data = []
    for country in countries:
        for year in years:
            data.append({
                'Country Name': country,
                'Year': year,
                'political_stability': np.random.randn(),
                'gdp_per_capita': np.random.uniform(1000, 50000),
                'gdp_growth': np.random.uniform(-2, 8),
                'unemployment_ilo': np.random.uniform(2, 15),
                'inflation_cpi': np.random.uniform(0, 10),
                'trade_gdp_pct': np.random.uniform(20, 100),
                'rule_of_law': np.random.uniform(-2, 2),
                'government_effectiveness': np.random.uniform(-2, 2),
                'hdi': np.random.uniform(0.4, 0.95)
            })

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(tmp_path, sample_panel_data):
    """
    Create temporary CSV file with sample data.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory
    sample_panel_data : pd.DataFrame
        Sample data

    Returns
    -------
    Path
        Path to temporary CSV file
    """
    file_path = tmp_path / "test_data.csv"
    sample_panel_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def logger():
    """Create logger for tests"""
    return logging.getLogger('test_panel')


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_panel_data_success(self, temp_data_file, logger):
        """Test successful data loading"""
        df = load_panel_data(temp_data_file, logger)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Country Name' in df.columns
        assert 'Year' in df.columns
        assert 'political_stability' in df.columns

    def test_load_panel_data_file_not_found(self, tmp_path, logger):
        """Test error when file doesn't exist"""
        non_existent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_panel_data(non_existent_file, logger)

    def test_load_panel_data_missing_columns(self, tmp_path, logger):
        """Test error when required columns are missing"""
        # Create file with missing columns
        df_incomplete = pd.DataFrame({
            'Country Name': ['A', 'B'],
            'Year': [2020, 2021]
            # Missing 'political_stability'
        })
        file_path = tmp_path / "incomplete.csv"
        df_incomplete.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_panel_data(file_path, logger)


class TestPanelStructure:
    """Tests for panel structure preparation"""

    def test_prepare_panel_structure_success(self, sample_panel_data, logger):
        """Test successful panel structure creation"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        assert isinstance(df_panel.index, pd.MultiIndex)
        assert df_panel.index.names == ['Country Name', 'Year']
        assert df_panel.index.is_monotonic_increasing

    def test_prepare_panel_structure_missing_entity_col(
        self, sample_panel_data, logger
    ):
        """Test error when entity column is missing"""
        with pytest.raises(ValueError, match="Entity column"):
            prepare_panel_structure(
                sample_panel_data,
                'NonExistentColumn',
                'Year',
                logger
            )

    def test_prepare_panel_structure_missing_time_col(
        self, sample_panel_data, logger
    ):
        """Test error when time column is missing"""
        with pytest.raises(ValueError, match="Time column"):
            prepare_panel_structure(
                sample_panel_data,
                'Country Name',
                'NonExistentColumn',
                logger
            )


class TestLaggedVariable:
    """Tests for lagged variable creation"""

    def test_create_lagged_variable(self, sample_panel_data, logger):
        """Test creation of lagged variable"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        df_lagged = create_lagged_variable(
            df_panel,
            'political_stability',
            1,
            logger
        )

        assert 'political_stability_lag1' in df_lagged.columns

        # Check lag is correct
        for country in sample_panel_data['Country Name'].unique():
            country_data = df_lagged.loc[country]
            # First year should be NaN
            assert pd.isna(country_data.iloc[0]['political_stability_lag1'])
            # Second year should equal first year's value
            if len(country_data) > 1:
                assert (
                    country_data.iloc[1]['political_stability_lag1'] ==
                    country_data.iloc[0]['political_stability']
                )


class TestModelFitting:
    """Tests for model fitting"""

    def test_fit_fixed_effects(self, sample_panel_data, logger):
        """Test Fixed Effects model fitting"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        predictors = ['gdp_per_capita', 'gdp_growth', 'rule_of_law']
        y = df_panel['political_stability']
        X = df_panel[predictors]

        results = fit_fixed_effects(y, X, logger)

        assert results is not None
        assert hasattr(results, 'params')
        assert hasattr(results, 'rsquared_within')
        assert len(results.params) == len(predictors)

    def test_fit_random_effects(self, sample_panel_data, logger):
        """Test Random Effects model fitting"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        predictors = ['gdp_per_capita', 'gdp_growth', 'rule_of_law']
        y = df_panel['political_stability']
        X = df_panel[predictors]

        results = fit_random_effects(y, X, logger)

        assert results is not None
        assert hasattr(results, 'params')
        assert len(results.params) == len(predictors)


class TestHausmanTest:
    """Tests for Hausman test"""

    def test_hausman_test(self, sample_panel_data, logger):
        """Test Hausman test execution"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        predictors = ['gdp_per_capita', 'gdp_growth', 'rule_of_law']
        y = df_panel['political_stability']
        X = df_panel[predictors]

        fe_results = fit_fixed_effects(y, X, logger)
        re_results = fit_random_effects(y, X, logger)

        hausman_result = hausman_test(fe_results, re_results, predictors, logger)

        assert isinstance(hausman_result, dict)
        assert 'statistic' in hausman_result
        assert 'p_value' in hausman_result
        assert 'decision' in hausman_result
        assert 'df' in hausman_result
        assert hausman_result['df'] == len(predictors)


class TestDynamicPanel:
    """Tests for dynamic panel analysis"""

    def test_fit_dynamic_panel(self, sample_panel_data, logger):
        """Test dynamic panel model fitting"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        df_dynamic = create_lagged_variable(
            df_panel,
            'political_stability',
            1,
            logger
        )
        df_dynamic = df_dynamic.dropna()

        predictors = ['gdp_per_capita', 'political_stability_lag1']
        y = df_dynamic['political_stability']
        X = df_dynamic[predictors]

        results = fit_dynamic_panel(y, X, logger)

        assert results is not None
        assert 'political_stability_lag1' in results.params.index

    def test_analyze_persistence(self, sample_panel_data, logger):
        """Test persistence analysis"""
        df_panel = prepare_panel_structure(
            sample_panel_data,
            'Country Name',
            'Year',
            logger
        )

        df_dynamic = create_lagged_variable(
            df_panel,
            'political_stability',
            1,
            logger
        )
        df_dynamic = df_dynamic.dropna()

        predictors = ['gdp_per_capita', 'political_stability_lag1']
        y = df_dynamic['political_stability']
        X = df_dynamic[predictors]

        results = fit_dynamic_panel(y, X, logger)
        persistence = analyze_persistence(results, 'political_stability', logger)

        assert isinstance(persistence, dict)
        assert 'coefficient' in persistence
        assert 'p_value' in persistence
        assert 'significant' in persistence


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_empty_dataframe(self, tmp_path, logger):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        file_path = tmp_path / "empty.csv"
        empty_df.to_csv(file_path, index=False)

        with pytest.raises(IOError, match="Failed to read CSV file"):
            load_panel_data(file_path, logger)

    def test_single_country(self, logger):
        """Test with single country (degenerate case)"""
        df_single = pd.DataFrame({
            'Country Name': ['CountryA'] * 5,
            'Year': [2018, 2019, 2020, 2021, 2022],
            'political_stability': np.random.randn(5),
            'gdp_per_capita': np.random.uniform(1000, 50000, 5),
            'rule_of_law': np.random.uniform(-2, 2, 5)
        })

        df_panel = prepare_panel_structure(
            df_single,
            'Country Name',
            'Year',
            logger
        )

        # Should still create valid panel structure
        assert isinstance(df_panel, pd.DataFrame)
        assert len(df_panel) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
