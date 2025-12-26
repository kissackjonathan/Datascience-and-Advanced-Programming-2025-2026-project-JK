"""
Data loading and preprocessing module

This module loads raw data files and prepares them for ML/econometric models.
Supports multiple file formats: CSV, Excel (.xlsx, .xls), Apple Numbers (.numbers)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: numbers-parser for Apple Numbers files
try:
    from numbers_parser import Document

    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False


def load_data_file(file_path: str, indicator_name: str) -> pd.DataFrame:
    """
    Load data from CSV/Excel/Numbers and convert World Bank wide format to long format.

    Parameters
    ----------
    file_path : str
        Path to the data file
    indicator_name : str
        Name for the indicator column

    Returns
    -------
    pd.DataFrame
        Long format with columns: Country Name, Year, indicator_name
    """
    file_path = Path(file_path)

    try:
        # Read file based on extension
        if file_path.suffix == ".numbers":
            if not NUMBERS_PARSER_AVAILABLE:
                return None
            # Read Numbers file (skip 4 rows for World Bank format)
            doc = Document(str(file_path))
            sheet = doc.sheets[0]
            table = sheet.tables[0]

            data = []
            for row_idx in range(4, table.num_rows):  # Skip 4 metadata rows
                row_data = []
                for col_idx in range(table.num_cols):
                    cell = table.cell(row_idx, col_idx)
                    row_data.append(cell.value)
                data.append(row_data)

            if not data:
                return None
            df = pd.DataFrame(data)
            if len(df) > 0:
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)

        elif file_path.suffix in [".xlsx", ".xls"]:
            # Try World Bank format (skip 4 rows)
            try:
                df = pd.read_excel(file_path, skiprows=4)
            except:
                df = pd.read_excel(file_path)

        elif file_path.suffix == ".csv":
            # Try World Bank format (skip 4 rows)
            try:
                df = pd.read_csv(file_path, skiprows=4)
            except:
                df = pd.read_csv(file_path)
        else:
            return None

    except Exception as e:
        return None

    # Clean and standardize
    df = df.dropna(axis=1, how="all")

    # Find Country Name column
    country_col_names = [
        "Country Name",
        "Country",
        "Country name",
        "country_name",
        "country",
    ]
    country_col = None
    for col in df.columns:
        if col in country_col_names:
            country_col = col
            break
    if country_col is None:
        country_col = df.columns[0]

    # Rename to standard name
    df = df.rename(columns={country_col: "Country Name"})

    # Keep Country Code if it exists (needed for map visualization)
    has_country_code = "Country Code" in df.columns

    # Drop only indicator metadata columns, but keep Country Code
    df = df.drop(columns=["Indicator Name", "Indicator Code"], errors="ignore")

    # Identify year columns (4-digit numbers between 1900-2100)
    year_cols = []
    for col in df.columns:
        if col not in ["Country Name", "Country Code", "Year"]:
            try:
                year = int(col)
                if 1900 <= year <= 2100:
                    year_cols.append(col)
            except:
                pass

    if not year_cols:
        # Maybe already in long format
        if "Year" in df.columns and indicator_name in df.columns:
            base_cols = ["Country Name", "Year", indicator_name]
            if has_country_code:
                base_cols.insert(1, "Country Code")
            return df[[c for c in base_cols if c in df.columns]]
        return None

    # Melt from wide to long format
    id_vars = ["Country Name"]
    if has_country_code:
        id_vars.append("Country Code")

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="Year",
        value_name=indicator_name,
    )

    # Convert Year to integer
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype("Int64")

    # Remove rows with missing Year
    df_long = df_long.dropna(subset=["Year"])

    return df_long


def load_data(
    data_path: Path, target: str = "political_stability", train_end_year: int = 2017
) -> Dict[str, pd.DataFrame]:
    """
    Load raw data files and prepare train/test splits.

    Pipeline:
    1. Load & Merge: World Bank indicators (wide→long format, OUTER join)
    2. Quality Filtering: UN whitelist, >30% missing removed (assure qualité du code)
    3. Imputation: Progressive window median fill (±1, ±2, ±4 years, global)
    4. Temporal Split: Train 1996-2017, Test 2018-2023
    5. Export: Save to data/processed/ for reproducibility

    Parameters
    ----------
    data_path : Path
        Path to data/raw/ directory
    target : str
        Target variable name
    train_end_year : int
        Last year for training set

    Returns
    -------
    dict
        Keys: 'X_train', 'X_test', 'y_train', 'y_test', 'df_train', 'df_test'
    """
    data_path = Path(data_path)

    # Define files to load with their indicator names
    files_to_load = [
        ("political stability", "political_stability"),  # TARGET VARIABLE
        ("GDP per capita", "gdp_per_capita"),
        ("UNEMPLOYMENT_TOTAL", "unemployment"),
        ("inflation consumer", "inflation"),
        ("GDP_GROWTH_%", "gdp_growth"),
        ("effectiveness", "effectiveness"),
        ("rule of law", "rule_of_law"),
        ("trade", "trade"),
    ]

    # Load each file
    dataframes = []

    for file_base, indicator in files_to_load:
        # Try multiple extensions
        for ext in [".csv", ".numbers", ".xlsx"]:
            file_path = data_path / f"{file_base}{ext}"
            if file_path.exists():
                df = load_data_file(str(file_path), indicator)
                if df is not None and len(df) > 0:
                    # Verify required columns exist
                    if "Country Name" in df.columns and "Year" in df.columns:
                        # Keep needed columns including Country Code if present
                        keep_cols = ["Country Name", "Year", indicator]
                        if "Country Code" in df.columns:
                            keep_cols.insert(1, "Country Code")
                        keep_cols = [c for c in keep_cols if c in df.columns]
                        df = df[keep_cols]
                        dataframes.append(df)
                break

    # Load HDI data (special handling for different format)
    hdi_file = data_path / "hdi_data.xlsx"
    if hdi_file.exists():
        try:
            hdi_raw = pd.read_excel(hdi_file)

            # HDI file structure: country, year, indexCode, value
            # Filter for HDI index only (exclude dimensions)
            if "indexCode" in hdi_raw.columns and "value" in hdi_raw.columns:
                hdi_raw = hdi_raw[hdi_raw["indexCode"] == "HDI"].copy()

                # Rename columns to match our format
                column_mapping = {
                    "country": "Country Name",
                    "year": "Year",
                    "value": "hdi",
                }
                hdi_raw = hdi_raw.rename(columns=column_mapping)

                # Keep only needed columns
                if (
                    "Country Name" in hdi_raw.columns
                    and "Year" in hdi_raw.columns
                    and "hdi" in hdi_raw.columns
                ):
                    hdi_df = hdi_raw[["Country Name", "Year", "hdi"]].copy()
                    # Convert Year to integer
                    hdi_df["Year"] = pd.to_numeric(
                        hdi_df["Year"], errors="coerce"
                    ).astype("Int64")
                    dataframes.append(hdi_df)
                    print(
                        f"DEBUG: HDI data loaded - {len(hdi_df)} rows, years {hdi_df['Year'].min()}-{hdi_df['Year'].max()}"
                    )
        except Exception as e:
            print(f"DEBUG: Failed to load HDI data: {e}")
            pass  # Skip HDI if loading fails

    if not dataframes:
        raise FileNotFoundError(f"No data files found in {data_path}")

    # =========================================================================
    # MERGE ALL INDICATORS
    # =========================================================================
    # - Different indicators may have data for different country-year combinations
    # - OUTER join preserves all observations (maximizes data availability)
    # - Missing values will be imputed later using median
    # - Alternative (INNER join) would only keep complete cases -> massive data loss
    # =========================================================================
    merged_df = dataframes[0]

    # Determine merge keys - include Country Code if present in all dataframes
    merge_keys = ["Country Name", "Year"]
    if all("Country Code" in df.columns for df in dataframes):
        merge_keys.insert(1, "Country Code")

    for df in dataframes[1:]:
        merged_df = merged_df.merge(
            df, on=merge_keys, how="outer", suffixes=("", "_dup")
        )

    # Drop duplicate columns
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_dup")]

    # Ensure target column exists (political_stability)
    if target not in merged_df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # CRITICAL: Convert all columns except Country Name and Country Code to numeric
    # Some columns might be loaded as object/string type
    for col in merged_df.columns:
        if col not in ["Country Name", "Country Code"]:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    # Sort by Country Name and Year to ensure proper time series order
    merged_df = merged_df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    print(
        f"DEBUG: Before filtering: {len(merged_df)} rows, {merged_df['Country Name'].nunique()} countries"
    )
    print(f"DEBUG: Columns present: {merged_df.columns.tolist()}")
    print(
        f"DEBUG: Numeric columns: {merged_df.select_dtypes(include=[np.number]).columns.tolist()}"
    )

    # SMART FILTERING: Remove countries with >30% missing FEATURE values
    # Calculate missing % per country only in rows that exist

    # First, drop rows with missing target
    merged_df = merged_df.dropna(subset=[target])
    print(f"DEBUG: After dropping target NaN: {len(merged_df)} rows")

    # =========================================================================
    # Country name normalization
    # Some countries are excluded because World Bank names differ from UN standards.
    # Solution: map World Bank country names to UN names before filtering.
    # =========================================================================

    COUNTRY_NAME_MAPPING = {
        # UN members with different names in World Bank data
        "Turkiye": "Turkey",
        "Viet Nam": "Vietnam",
        "Somalia, Fed. Rep.": "Somalia",
        # Additional territories with suffixes
        "Puerto Rico (US)": "Puerto Rico",
    }
    # Normalize country names before applying the filter
    countries_before_mapping = merged_df["Country Name"].nunique()
    merged_df["Country Name"] = merged_df["Country Name"].replace(COUNTRY_NAME_MAPPING)
    countries_after_mapping = merged_df["Country Name"].nunique()

    mapped_count = sum(
        1
        for k in COUNTRY_NAME_MAPPING.keys()
        if k in merged_df["Country Name"].unique()
    )
    print(f"DEBUG: Name normalization: {len(COUNTRY_NAME_MAPPING)} mappings defined")
    if mapped_count > 0:
        print(f"  - Renamed countries: {', '.join(COUNTRY_NAME_MAPPING.keys())}")

    # =========================================================================
    # Keep only UN members + selected territories (whitelist approach)
    # =========================================================================
    # Includes: 193 UN member states (2024) + 7 territories with strong/usable economic data
    # (Taiwan, Kosovo, Palestine, Hong Kong SAR, Macao SAR, Puerto Rico, Guam).
    # This automatically excludes World Bank aggregates (regions, income groups, unions, etc.).
    # =========================================================================

    # 193 UN member states (2024), using World Bank naming
    UN_MEMBER_STATES = [
        # A
        "Afghanistan",
        "Albania",
        "Algeria",
        "Andorra",
        "Angola",
        "Antigua and Barbuda",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        # B
        "Bahamas, The",
        "Bahrain",
        "Bangladesh",
        "Barbados",
        "Belarus",
        "Belgium",
        "Belize",
        "Benin",
        "Bhutan",
        "Bolivia",
        "Bosnia and Herzegovina",
        "Botswana",
        "Brazil",
        "Brunei Darussalam",
        "Bulgaria",
        "Burkina Faso",
        "Burundi",
        # C
        "Cabo Verde",
        "Cambodia",
        "Cameroon",
        "Canada",
        "Central African Republic",
        "Chad",
        "Chile",
        "China",
        "Colombia",
        "Comoros",
        "Congo, Dem. Rep.",
        "Congo, Rep.",
        "Costa Rica",
        "Cote d'Ivoire",
        "Croatia",
        "Cuba",
        "Cyprus",
        "Czechia",
        # D
        "Denmark",
        "Djibouti",
        "Dominica",
        "Dominican Republic",
        # E
        "Ecuador",
        "Egypt, Arab Rep.",
        "El Salvador",
        "Equatorial Guinea",
        "Eritrea",
        "Estonia",
        "Eswatini",
        "Ethiopia",
        # F
        "Fiji",
        "Finland",
        "France",
        # G
        "Gabon",
        "Gambia, The",
        "Georgia",
        "Germany",
        "Ghana",
        "Greece",
        "Grenada",
        "Guatemala",
        "Guinea",
        "Guinea-Bissau",
        "Guyana",
        # H
        "Haiti",
        "Honduras",
        "Hungary",
        # I
        "Iceland",
        "India",
        "Indonesia",
        "Iran, Islamic Rep.",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        # J
        "Jamaica",
        "Japan",
        "Jordan",
        # K
        "Kazakhstan",
        "Kenya",
        "Kiribati",
        "Korea, Dem. People's Rep.",
        "Korea, Rep.",
        "Kuwait",
        "Kyrgyz Republic",
        # L
        "Lao PDR",
        "Latvia",
        "Lebanon",
        "Lesotho",
        "Liberia",
        "Libya",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        # M
        "Madagascar",
        "Malawi",
        "Malaysia",
        "Maldives",
        "Mali",
        "Malta",
        "Marshall Islands",
        "Mauritania",
        "Mauritius",
        "Mexico",
        "Micronesia, Fed. Sts.",
        "Moldova",
        "Monaco",
        "Mongolia",
        "Montenegro",
        "Morocco",
        "Mozambique",
        "Myanmar",
        # N
        "Namibia",
        "Nauru",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nicaragua",
        "Niger",
        "Nigeria",
        "North Macedonia",
        "Norway",
        # O
        "Oman",
        # P
        "Pakistan",
        "Palau",
        "Panama",
        "Papua New Guinea",
        "Paraguay",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        # Q
        "Qatar",
        # R
        "Romania",
        "Russian Federation",
        "Rwanda",
        # S
        "Samoa",
        "San Marino",
        "Sao Tome and Principe",
        "Saudi Arabia",
        "Senegal",
        "Serbia",
        "Seychelles",
        "Sierra Leone",
        "Singapore",
        "Slovak Republic",
        "Slovenia",
        "Solomon Islands",
        "Somalia",
        "South Africa",
        "South Sudan",
        "Spain",
        "Sri Lanka",
        "St. Kitts and Nevis",
        "St. Lucia",
        "St. Vincent and the Grenadines",
        "Sudan",
        "Suriname",
        "Sweden",
        "Switzerland",
        "Syrian Arab Republic",
        # T
        "Tajikistan",
        "Tanzania",
        "Thailand",
        "Timor-Leste",
        "Togo",
        "Tonga",
        "Trinidad and Tobago",
        "Tunisia",
        "Turkey",
        "Turkmenistan",
        "Tuvalu",
        # U
        "Uganda",
        "Ukraine",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
        "Uruguay",
        "Uzbekistan",
        # V
        "Vanuatu",
        "Venezuela, RB",
        "Vietnam",
        # Y
        "Yemen, Rep.",
        # Z
        "Zambia",
        "Zimbabwe",
    ]
    # Additional territories with strong/usable economic data
    ADDITIONAL_TERRITORIES = [
        "Taiwan, China",
        "Kosovo",
        "West Bank and Gaza",
        "Hong Kong SAR, China",
        "Macao SAR, China",
        "Puerto Rico",
        "Guam",
    ]

    # Complete list of countries/territories to keep
    COUNTRIES_TO_KEEP = UN_MEMBER_STATES + ADDITIONAL_TERRITORIES

    # Count how many entities will be removed
    countries_before = merged_df["Country Name"].unique()
    countries_removed = [c for c in countries_before if c not in COUNTRIES_TO_KEEP]

    # Filter: keep only countries in the whitelist
    merged_df = merged_df[merged_df["Country Name"].isin(COUNTRIES_TO_KEEP)]

    print(f"DEBUG: Whitelist filter applied:")
    print(f"  - Membres ONU: {len(UN_MEMBER_STATES)}")
    print(f"  - Territoires additionnels: {len(ADDITIONAL_TERRITORIES)}")
    print(f"  - Total allowed: {len(COUNTRIES_TO_KEEP)}")
    print(f"  - Entities removed: {len(countries_removed)}")
    if len(countries_removed) > 0:
        print(f"  - Examples removed: {', '.join(list(countries_removed)[:5])}")
    print(
        f"DEBUG: After whitelist filtering: {len(merged_df)} rows, {merged_df['Country Name'].nunique()} countries"
    )

    # =========================================================================
    # FILTER COUNTRIES BY DATA QUALITY (>30% missing features)
    # =========================================================================
    # - Countries with too many missing values reduce model accuracy
    # - Imputation becomes unreliable when >30% values are missing
    # =========================================================================

    # Get feature columns (all numeric except target)
    all_numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in all_numeric_cols if col != target]

    countries_to_keep = []
    countries_eliminated = []

    for country in merged_df["Country Name"].unique():
        country_data = merged_df[merged_df["Country Name"] == country]

        # Calculate missing % only in EXISTING rows for FEATURE columns
        if len(feature_cols) > 0:
            total_values = len(country_data) * len(feature_cols)
            missing_values = country_data[feature_cols].isnull().sum().sum()
            missing_percentage = missing_values / total_values

            # Keep countries with <= 30% missing values
            if missing_percentage <= 0.30:
                countries_to_keep.append(country)
            else:
                countries_eliminated.append((country, missing_percentage))
        else:
            countries_to_keep.append(country)

    print(
        f"DEBUG: Removed countries (>30% missing features): {len(countries_eliminated)}"
    )

    # Filter to keep only good countries
    merged_df = merged_df[merged_df["Country Name"].isin(countries_to_keep)]
    print(
        f"DEBUG: After filter 30%: {len(merged_df)} rows, {len(countries_to_keep)} countries kept"
    )

    # ============================================================================
    # MISSING VALUE IMPUTATION - Unified Progressive Window Median Strategy
    # ============================================================================
    # Medians are applied sequentially; if not computable at one stage, proceed to the next
    # 1) Median within +/-1 year
    # 2) Median within +/-2 years
    # 3) Median within +/-4 years
    # 4) Global median
    #
    # Benefits: interpolates instead of copying, reduces artificial plateaus,
    # stays fully consistent (median-based), and better captures local trends.
    # ============================================================================

    # Get list of numeric feature columns (exclude target)
    numeric_cols = [
        col
        for col in merged_df.select_dtypes(include=[np.number]).columns
        if col != target
    ]

    print(f"DEBUG: Columns to fill: {numeric_cols}")
    print(f"DEBUG: NaN per column BEFORE filling:")
    for col in numeric_cols:
        nan_count = merged_df[col].isnull().sum()
        if nan_count > 0:
            print(f"  - {col}: {nan_count} NaN")

    print(f"DEBUG: UNIFIED Progressive Window Median Fill ")

    for col in numeric_cols:
        if not merged_df[col].isnull().any():
            continue  # Skip if no NaN in this column

        filled_window_1 = 0
        filled_window_2 = 0
        filled_window_4 = 0
        filled_global = 0

        # Get all rows with NaN in this column
        nan_mask = merged_df[col].isnull()
        nan_rows = merged_df[nan_mask]

        for idx in nan_rows.index:
            if pd.isnull(merged_df.loc[idx, col]):  # Check if still NaN
                country = merged_df.loc[idx, "Country Name"]
                year = merged_df.loc[idx, "Year"]

                # Get country data
                country_mask = merged_df["Country Name"] == country
                country_data = merged_df[country_mask]

                # STAGE 1: Try window ±1 year (very local interpolation)
                year_mask_1 = (country_data["Year"] >= year - 1) & (
                    country_data["Year"] <= year + 1
                )
                window_1_values = country_data[year_mask_1][col].dropna()

                if len(window_1_values) >= 2:  # Need at least 2 values for median
                    median_1 = window_1_values.median()
                    merged_df.loc[idx, col] = median_1
                    filled_window_1 += 1
                    continue

                # STAGE 2: Try window ±2 years
                year_mask_2 = (country_data["Year"] >= year - 2) & (
                    country_data["Year"] <= year + 2
                )
                window_2_values = country_data[year_mask_2][col].dropna()

                if len(window_2_values) >= 2:
                    median_2 = window_2_values.median()
                    merged_df.loc[idx, col] = median_2
                    filled_window_2 += 1
                    continue

                # STAGE 3: Try window ±4 years
                year_mask_4 = (country_data["Year"] >= year - 4) & (
                    country_data["Year"] <= year + 4
                )
                window_4_values = country_data[year_mask_4][col].dropna()

                if len(window_4_values) >= 2:
                    median_4 = window_4_values.median()
                    merged_df.loc[idx, col] = median_4
                    filled_window_4 += 1
                    continue

                # STAGE 4: Use global median as last resort
                global_median = merged_df[col].median()
                merged_df.loc[idx, col] = global_median
                filled_global += 1

        # Report statistics
        if filled_window_1 + filled_window_2 + filled_window_4 + filled_global > 0:
            print(f"  {col}:")
            if filled_window_1 > 0:
                print(
                    f"    - Window ±1 year:  {filled_window_1} values (interpolation)"
                )
            if filled_window_2 > 0:
                print(f"    - Window ±2 years: {filled_window_2} values")
            if filled_window_4 > 0:
                print(f"    - Window ±4 years: {filled_window_4} values")
            if filled_global > 0:
                print(f"    - Global median:   {filled_global} values")

    print(
        f"DEBUG: After unified median fill: {merged_df[numeric_cols].isnull().sum().sum()} NaN remaining"
    )
    print(
        f"DEBUG: FINAL - Rows: {len(merged_df)}, Countries: {merged_df['Country Name'].nunique()}, NaN: {merged_df.isnull().sum().sum()}"
    )

    # NO MORE DROPNA - we've filled everything
    # merged_df = merged_df.dropna()  # REMOVED - not needed anymore

    # Set multi-index (Country Name, Year) - include Country Code if present
    index_cols = ["Country Name", "Year"]
    if "Country Code" in merged_df.columns:
        index_cols.insert(1, "Country Code")

    if all(col in merged_df.columns for col in ["Country Name", "Year"]):
        merged_df = merged_df.set_index(index_cols)

    # Split into train/test by year
    train_df, test_df = get_train_test_split(merged_df.reset_index(), train_end_year)

    # Re-set index after split
    if all(col in train_df.columns for col in ["Country Name", "Year"]):
        # Use same index columns as before
        train_df = train_df.set_index(index_cols)
        test_df = test_df.set_index(index_cols)

    # Separate features (X) and target (y)
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # SAVE PROCESSED DATA to data/processed/
    # This allows models to load prepared data without re-processing raw files
    processed_dir = data_path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save train, test, and full data
    train_df.to_csv(processed_dir / "train_data.csv")
    test_df.to_csv(processed_dir / "test_data.csv")

    # Save full merged data (before split) for reference
    full_data = pd.concat([train_df, test_df])
    full_data.to_csv(processed_dir / "full_data.csv")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "df_train": train_df,
        "df_test": test_df,
    }


def get_train_test_split(
    df: pd.DataFrame, train_end_year: int = 2017
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split panel data by year (temporal split, not random)
    # Train: years <= train_end_year | Test: years > train_end_year
    df_train = df[df["Year"] <= train_end_year].copy()
    df_test = df[df["Year"] > train_end_year].copy()

    return df_train, df_test
