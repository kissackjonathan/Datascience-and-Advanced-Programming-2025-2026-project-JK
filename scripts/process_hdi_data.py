"""
Process UNDP HDI Excel data
"""
import pandas as pd
from pathlib import Path

# File path
file_path = "/Users/kissack.jonathan/Downloads/HDR25_Statistical_Annex_HDI_Table.xlsx"

# Read Excel with no header first
df_raw = pd.read_excel(file_path, sheet_name=0, header=None)

# Extract column names from row 5 (index 5)
col_names = df_raw.iloc[5].values

# Extract years from row 6 (index 6)
years = df_raw.iloc[6].values

# Read actual data starting from row 8 (skiprows=7)
df = pd.read_excel(file_path, sheet_name=0, skiprows=7, header=None)

# Set column names
df.columns = col_names

# Remove rows that are section headers (have NaN in rank column)
df = df[df.iloc[:, 0].notna()].copy()

# Select relevant columns: Rank, Country, HDI Value
# Column 0: HDI rank
# Column 1: Country
# Column 2: HDI Value (2023)

df_clean = df.iloc[:, [1, 2]].copy()
df_clean.columns = ['Country Name', 'hdi']

# Add year
df_clean['Year'] = 2023

# Remove any rows with missing values
df_clean = df_clean.dropna()

# Convert HDI to float
df_clean['hdi'] = pd.to_numeric(df_clean['hdi'], errors='coerce')
df_clean = df_clean.dropna(subset=['hdi'])

# Save
output_dir = Path(__file__).parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "undp_hdi.csv"
df_clean.to_csv(output_file, index=False)

print("=" * 70)
print("HDI DATA PROCESSING")
print("=" * 70)
print(f"Observations: {len(df_clean)}")
print(f"Year: 2023")
print(f"Countries: {df_clean['Country Name'].nunique()}")
print(f"HDI range: {df_clean['hdi'].min():.3f} - {df_clean['hdi'].max():.3f}")
print()
print(f"Sample (top 10):")
print(df_clean.head(10).to_string(index=False))
print()
print(f"Saved to: {output_file}")
print("=" * 70)
