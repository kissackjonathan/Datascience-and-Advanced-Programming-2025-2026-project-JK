"""
World Bank Data Loader
Load and parse data files from data/raw/ (CSV, Excel, Numbers)
NO API calls - uses existing downloaded files
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import glob
import warnings

try:
    from numbers_parser import Document
    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False
    print("Warning: numbers-parser not available. Cannot read .numbers files.")


def load_numbers_file(numbers_path: str, skip_rows: int = 0) -> pd.DataFrame:
    """
    Load data from Apple Numbers file (.numbers)

    Uses the numbers-parser library to read .numbers files directly.

    Args:
        numbers_path: Path to the .numbers file
        skip_rows: Number of rows to skip (default 0)

    Returns:
        DataFrame with the extracted data
    """
    if not NUMBERS_PARSER_AVAILABLE:
        raise ValueError("numbers-parser package not installed. Run: pip install numbers-parser")

    try:
        # Open the Numbers document
        doc = Document(numbers_path)

        # Get the first sheet
        sheets = doc.sheets
        if not sheets:
            raise ValueError(f"No sheets found in {numbers_path}")

        sheet = sheets[0]

        # Get the first table from the sheet
        tables = sheet.tables
        if not tables:
            raise ValueError(f"No tables found in {numbers_path}")

        table = tables[0]

        # Convert table to pandas DataFrame
        data = []
        num_rows = table.num_rows
        num_cols = table.num_cols

        # Skip the specified number of rows
        for row_idx in range(skip_rows, num_rows):
            row_data = []
            for col_idx in range(num_cols):
                cell = table.cell(row_idx, col_idx)
                row_data.append(cell.value)
            data.append(row_data)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Use first row as header
        if len(df) > 0:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

        return df

    except Exception as e:
        raise ValueError(f"Could not read Numbers file {numbers_path}: {e}")


def load_data_file(file_path: str, indicator_name: str) -> pd.DataFrame:
    """
    Load data from various file formats (CSV, Excel, Numbers)

    Handles different formats and structures:
    - World Bank API CSV format (skip 4 rows, wide format with years as columns)
    - Simple CSV/Excel with standard structure
    - Apple Numbers files (extract from ZIP archive)

    Args:
        file_path: Path to the data file
        indicator_name: Name to use for the indicator column

    Returns:
        DataFrame with columns: Country Name, Country Code, Year, indicator_name
    """
    file_path = Path(file_path)

    # Try to read the file based on extension
    try:
        if file_path.suffix == '.numbers':
            # World Bank .numbers files have 4 metadata rows to skip
            df = load_numbers_file(str(file_path), skip_rows=4)
        elif file_path.suffix in ['.xlsx', '.xls']:
            # Try World Bank format first (skip 4 rows)
            try:
                df = pd.read_excel(file_path, skiprows=4)
            except:
                df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            # Try World Bank format first (skip 4 rows)
            try:
                df = pd.read_csv(file_path, skiprows=4)
            except:
                df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Clean and standardize the dataframe
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')

    # Try to identify Country Name and Country Code columns
    # Look for common variations
    country_col_names = ['Country Name', 'Country', 'Country name', 'country_name', 'country']
    code_col_names = ['Country Code', 'Code', 'Country code', 'country_code', 'code']

    country_col = None
    code_col = None

    for col in df.columns:
        if col in country_col_names:
            country_col = col
        if col in code_col_names:
            code_col = col

    # If not found, use first two columns as country and code
    if country_col is None:
        country_col = df.columns[0]
    if code_col is None and len(df.columns) > 1:
        code_col = df.columns[1]

    # Rename to standard names
    df = df.rename(columns={country_col: 'Country Name'})
    if code_col:
        df = df.rename(columns={code_col: 'Country Code'})

    # Drop metadata columns if present
    df = df.drop(columns=['Indicator Name', 'Indicator Code'], errors='ignore')

    # Identify year columns (numeric column names or columns that look like years)
    year_cols = []
    for col in df.columns:
        if col not in ['Country Name', 'Country Code', 'Year']:
            # Check if column name is a year (4-digit number)
            try:
                year = int(col)
                if 1900 <= year <= 2100:
                    year_cols.append(col)
            except:
                pass

    if not year_cols:
        # Maybe the data is already in long format
        if 'Year' in df.columns and indicator_name in df.columns:
            return df[['Country Name', 'Country Code', 'Year', indicator_name]] if 'Country Code' in df.columns else df[['Country Name', 'Year', indicator_name]]

        # Otherwise, assume all non-Country columns are data
        year_cols = [col for col in df.columns if col not in ['Country Name', 'Country Code']]

    # Melt from wide to long format
    id_vars = ['Country Name']
    if 'Country Code' in df.columns:
        id_vars.append('Country Code')

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name=indicator_name
    )

    # Convert Year to integer
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')

    # Remove rows with missing Year or missing values
    df_long = df_long.dropna(subset=['Year'])

    return df_long


def load_all_wb_data(raw_dir: str = 'data/raw') -> Dict[str, pd.DataFrame]:
    """
    Load all World Bank indicators from data/raw/

    Supports multiple file formats: .numbers, .xlsx, .csv
    Tries multiple possible file names for each indicator

    Indicators loaded:
    - 6 economic: GDP per capita, GDP growth, unemployment, inflation, Gini, trade
    - 4 WGI governance: political stability, rule of law, govt effectiveness, control of corruption

    Args:
        raw_dir: Directory containing data files (default: 'data/raw')

    Returns:
        Dictionary mapping indicator names to DataFrames
    """
    raw_path = Path(raw_dir)

    # Map indicator names to possible file names (try multiple variations and extensions)
    indicators = {
        'gdp_per_capita': ['GDP per capita', 'gdp_per_capita', 'API_NY.GDP.PCAP.CD_*'],
        'gdp_growth': ['GDP_GROWTH_%', 'gdp_growth', 'GDP growth', 'API_NY.GDP.MKTP.KD.ZG_*'],
        'unemployment_ilo': ['UNEMPLOYMENT_TOTAL', 'unemployment_ilo', 'unemployment', 'API_SL.UEM.TOTL.NE.ZS_*'],
        'inflation_cpi': ['inflation consumer', 'inflation_cpi', 'inflation', 'API_FP.CPI.TOTL.ZG_*'],
        'gini_index': ['gini index', 'gini_index', 'gini', 'API_SI.POV.GINI_*'],
        'trade_gdp_pct': ['trade', 'trade_gdp_pct', 'API_NE.TRD.GNFS.ZS_*'],
        'political_stability': ['political stability', 'political_stability', 'API_PV.EST_*'],
        'rule_of_law': ['rule of law', 'rule_of_law', 'API_RL.EST_*'],
        'government_effectiveness': ['effectiveness', 'government_effectiveness', 'govt_effectiveness', 'API_GE.EST_*'],
        'control_of_corruption': ['control of corruption', 'control_of_corruption', 'API_CC.EST_*'],
    }

    # Extensions to try (in order of preference)
    extensions = ['.numbers', '.xlsx', '.xls', '.csv']

    data_dict = {}

    for indicator_name, possible_names in indicators.items():
        file_found = None

        # Try each possible name with each extension
        for name in possible_names:
            if file_found:
                break

            # If name contains wildcard, use glob
            if '*' in name:
                matches = list(raw_path.glob(name))
                if matches:
                    file_found = matches[0]
                    break
            else:
                # Try each extension
                for ext in extensions:
                    file_path = raw_path / f"{name}{ext}"
                    if file_path.exists():
                        file_found = file_path
                        break

        if not file_found:
            print(f"⚠️  Warning: No file found for {indicator_name}")
            print(f"    Tried: {', '.join(possible_names)}")
            continue

        print(f"✓ Loading {indicator_name} from {file_found.name}")

        # Load and parse file
        df = load_data_file(str(file_found), indicator_name)

        if df is not None:
            data_dict[indicator_name] = df
        else:
            print(f"⚠️  Failed to load {indicator_name} from {file_found.name}")

    print(f"\n✓ Loaded {len(data_dict)}/{len(indicators)} indicators")
    return data_dict


def save_processed_wb_data(data_dict: Dict[str, pd.DataFrame], output_dir: str = 'data/processed') -> None:
    """
    Save processed World Bank data to separate CSV files

    Args:
        data_dict: Dictionary mapping indicator names to DataFrames
        output_dir: Directory to save processed files (default: 'data/processed')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for indicator_name, df in data_dict.items():
        output_file = output_path / f'wb_{indicator_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Saved {indicator_name} → {output_file.name} ({len(df)} rows)")

    print(f"\n✓ All files saved to {output_dir}/")


def load_undp_hdi(hdi_file: str = 'data/raw/hdi_data.xlsx', harmonize: bool = True) -> pd.DataFrame:
    """
    Load UNDP Human Development Index (HDI) data

    HDI data is already in long format (one row per country-year).

    Args:
        hdi_file: Path to the HDI Excel file (default: 'data/raw/hdi_data.xlsx')
        harmonize: If True, harmonize country names to match World Bank format (default: True)

    Returns:
        DataFrame with columns: Country Name, Country Code, Year, hdi
    """
    try:
        # Load HDI data
        df = pd.read_excel(hdi_file)

        # Filter to HDI index only (in case file contains other indices)
        if 'indexCode' in df.columns:
            df = df[df['indexCode'] == 'HDI'].copy()

        # Select and rename relevant columns
        df = df[['country', 'countryIsoCode', 'year', 'value']].copy()
        df = df.rename(columns={
            'country': 'Country Name',
            'countryIsoCode': 'Country Code',
            'year': 'Year',
            'value': 'hdi'
        })

        # Clean data
        df = df.dropna(subset=['hdi'])
        df['Year'] = df['Year'].astype('Int64')

        # Harmonize country names to match World Bank format
        if harmonize:
            df = harmonize_country_names(df, source='undp')
            print(f"✓ Harmonized country names to World Bank format")

        print(f"✓ Loaded HDI data: {len(df)} observations")
        print(f"  Countries: {df['Country Name'].nunique()}")
        print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")

        return df

    except Exception as e:
        print(f"⚠️  Error loading HDI data: {e}")
        return None


def save_undp_hdi(df: pd.DataFrame, output_dir: str = 'data/processed') -> None:
    """
    Save processed HDI data to CSV

    Args:
        df: HDI DataFrame
        output_dir: Directory to save processed file (default: 'data/processed')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'undp_hdi.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Saved HDI → {output_file.name} ({len(df)} rows)")


def harmonize_country_names(df: pd.DataFrame, source: str = 'undp') -> pd.DataFrame:
    """
    Harmonize country names between UNDP and World Bank datasets

    Args:
        df: DataFrame with 'Country Name' column
        source: 'undp' to convert UNDP names to WB format, 'wb' for reverse

    Returns:
        DataFrame with harmonized country names
    """
    # Mapping from UNDP names to World Bank names
    undp_to_wb = {
        'Bahamas': 'Bahamas, The',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Congo': 'Congo, Rep.',
        'Congo (Democratic Republic of the)': 'Congo, Dem. Rep.',
        'Côte d\'Ivoire': 'Cote d\'Ivoire',
        'Egypt': 'Egypt, Arab Rep.',
        'Eswatini (Kingdom of)': 'Eswatini',
        'Gambia': 'Gambia, The',
        'Hong Kong, China (SAR)': 'Hong Kong SAR, China',
        'Iran (Islamic Republic of)': 'Iran, Islamic Rep.',
        'Korea (Republic of)': 'Korea, Rep.',
        'Kyrgyzstan': 'Kyrgyz Republic',
        'Lao People\'s Democratic Republic': 'Lao PDR',
        'Micronesia (Federated States of)': 'Micronesia, Fed. Sts.',
        'Moldova (Republic of)': 'Moldova',
        'Palestine, State of': 'West Bank and Gaza',
        'Saint Kitts and Nevis': 'St. Kitts and Nevis',
        'Saint Lucia': 'St. Lucia',
        'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
        'Slovakia': 'Slovak Republic',
        'Somalia': 'Somalia, Fed. Rep.',
        'Tanzania (United Republic of)': 'Tanzania',
        'Türkiye': 'Turkiye',
        'Venezuela (Bolivarian Republic of)': 'Venezuela, RB',
        'Yemen': 'Yemen, Rep.',
    }

    # Create reverse mapping for WB to UNDP
    wb_to_undp = {v: k for k, v in undp_to_wb.items()}

    df_harmonized = df.copy()

    if source == 'undp':
        # Convert UNDP names to WB format
        df_harmonized['Country Name'] = df_harmonized['Country Name'].replace(undp_to_wb)
    elif source == 'wb':
        # Convert WB names to UNDP format
        df_harmonized['Country Name'] = df_harmonized['Country Name'].replace(wb_to_undp)
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'undp' or 'wb'")

    return df_harmonized


def merge_all_data(processed_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Merge all processed data (World Bank + UNDP) into a single DataFrame

    Loads all individual CSV files and merges them on Country Name and Year.

    Args:
        processed_dir: Directory containing processed CSV files (default: 'data/processed')

    Returns:
        DataFrame with all variables merged
    """
    processed_path = Path(processed_dir)

    # Define all data files to merge
    data_files = {
        # Economic variables
        'gdp_per_capita': 'wb_gdp_per_capita.csv',
        'gdp_growth': 'wb_gdp_growth.csv',
        'unemployment_ilo': 'wb_unemployment_ilo.csv',
        'inflation_cpi': 'wb_inflation_cpi.csv',
        'gini_index': 'wb_gini_index.csv',
        'trade_gdp_pct': 'wb_trade_gdp_pct.csv',
        # WGI governance variables
        'rule_of_law': 'wb_rule_of_law.csv',
        'government_effectiveness': 'wb_government_effectiveness.csv',
        # Target variable
        'political_stability': 'wb_political_stability.csv',
        # UNDP HDI
        'hdi': 'undp_hdi.csv'
    }

    print("Merging all data files...")
    print()

    # Start with political stability (target variable)
    df_merged = pd.read_csv(processed_path / data_files['political_stability'])
    print(f"✓ Starting with political_stability: {len(df_merged)} rows")

    # Merge all other variables
    for var_name, filename in data_files.items():
        if var_name == 'political_stability':
            continue  # Already loaded

        file_path = processed_path / filename
        if not file_path.exists():
            print(f"⚠️  Warning: {filename} not found, skipping")
            continue

        df_temp = pd.read_csv(file_path)

        # Merge on Country Name and Year
        before_merge = len(df_merged)
        df_merged = df_merged.merge(
            df_temp[['Country Name', 'Year', var_name]],
            on=['Country Name', 'Year'],
            how='left'
        )
        print(f"✓ Merged {var_name}: {len(df_temp)} rows → {len(df_merged)} rows after merge")

    print()
    print(f"✓ Final merged data: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    print(f"  Countries: {df_merged['Country Name'].nunique()}")
    print(f"  Years: {df_merged['Year'].min():.0f} - {df_merged['Year'].max():.0f}")

    # Show missing data summary
    print()
    print("Missing data summary:")
    for col in df_merged.columns:
        if col not in ['Country Name', 'Country Code', 'Year']:
            missing_pct = df_merged[col].isnull().sum() / len(df_merged) * 100
            print(f"  {col:30s}: {missing_pct:>6.2f}% missing")

    return df_merged


def save_merged_data(df: pd.DataFrame, output_dir: str = 'data/processed', filename: str = 'merged_data.csv') -> None:
    """
    Save merged data to CSV

    Args:
        df: Merged DataFrame
        output_dir: Directory to save file (default: 'data/processed')
        filename: Output filename (default: 'merged_data.csv')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / filename
    df.to_csv(output_file, index=False)
    print()
    print(f"✓ Saved merged data → {output_file.name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    """
    Test script to run the data acquisition pipeline
    """
    print("=" * 80)
    print("WORLD BANK DATA LOADER")
    print("=" * 80)
    print()

    # Load all World Bank data
    data = load_all_wb_data(raw_dir='data/raw')

    # Save processed World Bank data
    save_processed_wb_data(data, output_dir='data/processed')

    print()
    print("=" * 80)
    print("UNDP HDI DATA LOADER")
    print("=" * 80)
    print()

    # Load HDI data
    hdi_df = load_undp_hdi(hdi_file='data/raw/hdi_data.xlsx')

    if hdi_df is not None:
        # Save processed HDI data
        save_undp_hdi(hdi_df, output_dir='data/processed')

    print()
    print("=" * 80)
    print("MERGING ALL DATA")
    print("=" * 80)
    print()

    # Merge all processed data into one DataFrame
    df_merged = merge_all_data(processed_dir='data/processed')

    # Save merged data
    save_merged_data(df_merged, output_dir='data/processed', filename='merged_data.csv')

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
