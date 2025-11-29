"""
Process World Bank CSV data from extracted folders
"""
import pandas as pd
from pathlib import Path


def process_wb_csv(csv_path, indicator_name):
    """Process World Bank CSV file"""
    try:
        # Read CSV (skip first 4 rows of metadata)
        df = pd.read_csv(csv_path, skiprows=4)

        # Get year columns
        id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        year_cols = [col for col in df.columns if col not in id_cols]

        # Reshape to long format
        df_long = df.melt(
            id_vars=['Country Name', 'Country Code'],
            value_vars=year_cols,
            var_name='Year',
            value_name=indicator_name
        )

        # Clean
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
        df_long = df_long.dropna(subset=['Year', indicator_name])
        df_long['Year'] = df_long['Year'].astype(int)

        return df_long

    except Exception as e:
        print(f"Error: {e}")
        return None


# File paths
files = {
    'gdp_per_capita': '/Users/kissack.jonathan/Downloads/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_267552/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_267552.csv',
    'gdp_growth': '/Users/kissack.jonathan/Downloads/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_260128/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_260128.csv',
    'unemployment_ilo': '/Users/kissack.jonathan/Downloads/API_SL.UEM.TOTL.NE.ZS_DS2_en_csv_v2_125709/API_SL.UEM.TOTL.NE.ZS_DS2_en_csv_v2_125709.csv',
    'unemployment_national': '/Users/kissack.jonathan/Downloads/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_254884/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_254884.csv',
    'inflation_cpi': '/Users/kissack.jonathan/Downloads/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_268992/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_268992.csv',
    'gini_index': '/Users/kissack.jonathan/Downloads/API_SI/API_SI.POV.GINI_DS2_en_csv_v2_252305.csv',
    'trade_gdp_pct': '/Users/kissack.jonathan/Downloads/API_NE/API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_242492.csv',
    'political_stability': '/Users/kissack.jonathan/Downloads/API_PV.EST_DS2_en_csv_v2_126909/API_PV.EST_DS2_en_csv_v2_126909.csv',
    # NEW: Worldwide Governance Indicators (WGI)
    'rule_of_law': '/Users/kissack.jonathan/Downloads/API_RL.EST_DS2_en_csv_v2_6627/API_RL.EST_DS2_en_csv_v2_6627.csv',
    'control_of_corruption': '/Users/kissack.jonathan/Downloads/API_CC.EST_DS2_en_csv_v2_5926/API_CC.EST_DS2_en_csv_v2_5926.csv',
    'government_effectiveness': '/Users/kissack.jonathan/Downloads/API_GE.EST_DS2_en_csv_v2_127143/API_GE.EST_DS2_en_csv_v2_127143.csv'
}

# Output directory
output_dir = Path(__file__).parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PROCESSING WORLD BANK DATA")
print("=" * 70)
print()

for name, path in files.items():
    print(f"Processing: {name}")

    if not Path(path).exists():
        print(f"  SKIP: File not found")
        print()
        continue

    df = process_wb_csv(path, name)

    if df is not None:
        # Save
        output_file = output_dir / f"wb_{name}.csv"
        df.to_csv(output_file, index=False)

        print(f"  {len(df)} observations")
        print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
        print(f"  Countries: {df['Country Name'].nunique()}")
        print(f"  Saved: {output_file.name}")

    print()

print("=" * 70)
print("DONE")
print(f"Files saved to: {output_dir}")
print("=" * 70)
