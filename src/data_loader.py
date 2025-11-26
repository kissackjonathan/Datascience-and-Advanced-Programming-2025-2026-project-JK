"""
Data loading and preprocessing module
Handles FSI, World Bank, and UNDP data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Load and preprocess data from multiple sources"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)

    def load_fsi_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load Fragile States Index data

        Args:
            filepath: Path to FSI data file

        Returns:
            DataFrame with FSI scores
        """
        if filepath is None:
            filepath = self.data_dir / "fsi_data.csv"

        # TODO: Implement FSI data loading
        raise NotImplementedError("FSI data loading to be implemented")

    def load_worldbank_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load World Bank economic indicators

        Args:
            filepath: Path to World Bank data file

        Returns:
            DataFrame with economic indicators
        """
        if filepath is None:
            filepath = self.data_dir / "worldbank_data.csv"

        # TODO: Implement World Bank data loading
        raise NotImplementedError("World Bank data loading to be implemented")

    def load_undp_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load UNDP Human Development Index data

        Args:
            filepath: Path to UNDP data file

        Returns:
            DataFrame with HDI components
        """
        if filepath is None:
            filepath = self.data_dir / "undp_data.csv"

        # TODO: Implement UNDP data loading
        raise NotImplementedError("UNDP data loading to be implemented")

    def merge_datasets(
        self,
        fsi: pd.DataFrame,
        worldbank: pd.DataFrame,
        undp: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all datasets on country and year

        Args:
            fsi: FSI dataframe
            worldbank: World Bank dataframe
            undp: UNDP dataframe

        Returns:
            Merged dataframe
        """
        # TODO: Implement merging logic
        raise NotImplementedError("Data merging to be implemented")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input dataframe

        Returns:
            Cleaned dataframe
        """
        # TODO: Implement missing value handling
        raise NotImplementedError("Missing value handling to be implemented")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features (lags, changes, interactions)

        Args:
            df: Input dataframe

        Returns:
            DataFrame with engineered features
        """
        # TODO: Implement feature engineering
        # - Lagged variables (FSI_t-1, GDP_t-1)
        # - Changes (Δ_GDP, Δ_Unemployment)
        # - Interactions (GDP × Education, Inflation × Debt)
        raise NotImplementedError("Feature engineering to be implemented")

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete pipeline: load, merge, clean, and engineer features

        Returns:
            Tuple of (X_features, y_target)
        """
        # TODO: Implement complete pipeline
        raise NotImplementedError("Complete pipeline to be implemented")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset

    Args:
        df: Input dataframe

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_countries': df['country'].nunique() if 'country' in df.columns else None,
        'year_range': (df['year'].min(), df['year'].max()) if 'year' in df.columns else None,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return summary
