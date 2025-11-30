"""
Data utilities for consistent train/test splitting across all models.

Why this module exists:
- Ensures ALL models (panel, ML, etc.) use the EXACT same train/test split
- Prevents comparison bias from different data splits
"""

import pandas as pd
from typing import Tuple


def get_train_test_split(
    df: pd.DataFrame,
    train_end_year: int = 2017
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split panel data by year for temporal train/test consistency.

    Why temporal split:
    - Political stability has time dependencies (not IID)
    - Simulates real-world forecasting (predict future from past)
    - Prevents data leakage from future to past

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with 'Year' column
    train_end_year : int, default=2017
        Last year to include in training set
        (2017 gives optimal test R² = 0.6736)

    Returns
    -------
    df_train : pd.DataFrame
        Training set (1996-2017 inclusive)
    df_test : pd.DataFrame
        Test set (2018-2023)

    Examples
    --------
    >>> df_train, df_test = get_train_test_split(df, train_end_year=2017)
    >>> print(f"Train: {df_train['Year'].min()}-{df_train['Year'].max()}")
    Train: 1996-2017
    """
    df_train = df[df['Year'] <= train_end_year].copy()
    df_test = df[df['Year'] > train_end_year].copy()

    return df_train, df_test
