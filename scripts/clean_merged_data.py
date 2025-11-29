"""
Clean Merged Data
- Filter to UN recognized countries only (193 countries)
- Filter to years >= 1996 (when political_stability starts)
- Analyze missing values
"""
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("CLEANING MERGED DATA")
print("=" * 80)
print()

# Load merged data
df = pd.read_csv(data_dir / "merged_data.csv")
print(f"Initial data: {len(df)} rows, {df['Country Name'].nunique()} countries")
print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Step 1: Filter to years >= 1996 (when political_stability starts)
print("Step 1: Filter to years >= 1996")
df = df[df['Year'] >= 1996].copy()
print(f"After year filter: {len(df)} rows")
print()

# Step 2: Filter to UN recognized countries only (exclude regional aggregates)
# Load HDI data to get list of valid countries (HDI contains only individual countries)
hdi_df = pd.read_csv(data_dir / "undp_hdi.csv")
valid_countries = hdi_df['Country Name'].unique()

print(f"Step 2: Filter to UN recognized countries")
print(f"Valid countries from HDI: {len(valid_countries)}")

# Show some examples of excluded countries (aggregates)
excluded = df[~df['Country Name'].isin(valid_countries)]['Country Name'].unique()
print(f"Excluded (regional aggregates): {len(excluded)} entities")
print("Examples of excluded:")
for country in sorted(excluded)[:10]:
    print(f"  - {country}")
print()

# Filter to valid countries
df = df[df['Country Name'].isin(valid_countries)].copy()
print(f"After country filter: {len(df)} rows, {df['Country Name'].nunique()} countries")
print()

# Step 3: Analyze missing values
print("=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)
print()

# Overall missing stats
print("Missing values per variable:")
print(f"{'Variable':<30} {'Missing':>8} {'Total':>8} {'Percent':>10}")
print("-" * 60)

variables = [col for col in df.columns if col not in ['Country Name', 'Country Code', 'Year']]

for var in variables:
    missing = df[var].isnull().sum()
    total = len(df)
    pct = missing / total * 100
    print(f"{var:<30} {missing:>8} {total:>8} {pct:>9.2f}%")

print()

# Missing by country
print("=" * 80)
print("COUNTRIES WITH MOST MISSING DATA")
print("=" * 80)
print()

country_missing = df.groupby('Country Name').agg({
    var: lambda x: x.isnull().sum() / len(x) * 100 for var in variables
})

# Average missing across all variables
country_missing['avg_missing'] = country_missing.mean(axis=1)
country_missing = country_missing.sort_values('avg_missing', ascending=False)

print("Top 20 countries with most missing data:")
print(f"{'Country':<35} {'Avg Missing':>12}")
print("-" * 50)
for country in country_missing.head(20).index:
    avg_missing = country_missing.loc[country, 'avg_missing']
    print(f"{country:<35} {avg_missing:>11.1f}%")

print()

# Countries with least missing
print("Top 20 countries with least missing data:")
print(f"{'Country':<35} {'Avg Missing':>12}")
print("-" * 50)
for country in country_missing.tail(20).index:
    avg_missing = country_missing.loc[country, 'avg_missing']
    print(f"{country:<35} {avg_missing:>11.1f}%")

print()

# Distribution of missing data
print("=" * 80)
print("DISTRIBUTION OF MISSING DATA")
print("=" * 80)
print()

thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(len(thresholds) - 1):
    low, high = thresholds[i], thresholds[i+1]
    count = ((country_missing['avg_missing'] >= low) & (country_missing['avg_missing'] < high)).sum()
    print(f"Countries with {low:>3}% - {high:>3}% missing: {count:>3}")

print()

# Complete data analysis
print("=" * 80)
print("COMPLETE DATA ANALYSIS")
print("=" * 80)
print()

# Rows with no missing values
complete_rows = df.dropna()
print(f"Rows with NO missing values: {len(complete_rows)} / {len(df)} ({len(complete_rows)/len(df)*100:.1f}%)")
print(f"Countries with at least 1 complete row: {complete_rows['Country Name'].nunique()}")

print()

# Countries with no complete rows
countries_no_complete = set(df['Country Name'].unique()) - set(complete_rows['Country Name'].unique())
print(f"Countries with NO complete rows: {len(countries_no_complete)}")
if len(countries_no_complete) > 0:
    print("Examples:")
    for country in sorted(list(countries_no_complete))[:20]:
        print(f"  - {country}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"After filtering:")
print(f"  - Years: 1996-2023")
print(f"  - Countries: {df['Country Name'].nunique()} (UN recognized)")
print(f"  - Total observations: {len(df)}")
print(f"  - Complete observations (no missing): {len(complete_rows)} ({len(complete_rows)/len(df)*100:.1f}%)")
print()

# Save cleaned data (filtered but not imputed)
output_file = data_dir / "filtered_data.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved filtered data → {output_file.name}")
print(f"  (Note: Missing values NOT yet imputed)")
print()
