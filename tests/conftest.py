"""
Shared pytest fixtures for all test modules.
"""
# Set matplotlib backend before any imports
import matplotlib

matplotlib.use("Agg")

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    n_train, n_test, n_features = 100, 30, 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_train = pd.Series(
        X_train["f0"] * 2 + X_train["f1"] * 1.5 + np.random.randn(n_train) * 0.2
    )

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_test = pd.Series(
        X_test["f0"] * 2 + X_test["f1"] * 1.5 + np.random.randn(n_test) * 0.2
    )

    return X_train, y_train, X_test, y_test


@pytest.fixture
def temp_dir():
    """Create a temporary directory for saving models/files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def panel_data():
    """Generate synthetic panel data for PanelAnalyzer tests."""
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(20)]
    years = list(range(2000, 2020))

    data = []
    for country in countries:
        country_effect = np.random.randn()
        for year in years:
            data.append(
                {
                    "Country Name": country,
                    "Year": year,
                    "political_stability": 0.5
                    + country_effect * 0.3
                    + np.random.randn() * 0.2,
                    "gdp_per_capita": 40000
                    + country_effect * 5000
                    + np.random.randn() * 2000,
                    "unemployment": 5 + np.random.randn() * 1.5,
                    "inflation": 2 + np.random.randn() * 0.8,
                    "gdp_growth": 2.5 + np.random.randn() * 1.2,
                    "effectiveness": 0.5
                    + country_effect * 0.2
                    + np.random.randn() * 0.15,
                    "rule_of_law": 0.5
                    + country_effect * 0.2
                    + np.random.randn() * 0.15,
                    "trade": 50 + np.random.randn() * 10,
                    "hdi": 0.7 + country_effect * 0.1 + np.random.randn() * 0.05,
                }
            )

    return pd.DataFrame(data)
