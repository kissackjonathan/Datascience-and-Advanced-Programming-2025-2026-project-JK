"""
Prepare Final Clean Data - OPTION 3 (HYBRID) - IMPROVED

Strategy:
1. Drop gini_index (49% missing - too sparse)
2. Keep all 193 UN countries
3. Filter to years >= 1996 (when political_stability starts)
4. Impute missing values with LINEAR INTERPOLATION (respects time series structure)
5. Keep: 8 predictors + 1 target = 9 variables total

Variables:
- Economic: gdp_per_capita, gdp_growth, unemployment_ilo, inflation_cpi, trade_gdp_pct
- Governance: rule_of_law, government_effectiveness
- Development: hdi
- Target: political_stability

IMPORTANT NOTE - MISSING YEARS (1997, 1999, 2001):
================================================================================
Les Worldwide Governance Indicators (WGI) de la Banque Mondiale, qui incluent
political_stability, rule_of_law, et government_effectiveness, n'ont PAS été
collectés annuellement pendant les premières années:

- 1996-2002: Collecte BISANNUELLE (tous les 2 ans)
  → Années disponibles: 1996, 1998, 2000, 2002
  → Années manquantes: 1997, 1999, 2001

- 2003-2023: Collecte ANNUELLE
  → Toutes les années sont disponibles

Résultat: Le dataset final contient 25 années sur la période 1996-2023
(28 ans - 3 ans manquants = 25 ans de données)

Ceci est NORMAL et reflète la réalité de la collecte des données WGI.
Ne pas interpréter comme un bug ou une erreur de traitement.
================================================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("PREPARING FINAL CLEAN DATA - OPTION 3 (HYBRID)")
print("=" * 80)
print()

# Load merged data
df = pd.read_csv(data_dir / "merged_data.csv")
print(f"Initial data: {len(df)} rows, {df['Country Name'].nunique()} countries")
print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Important note about missing years
print("NOTE IMPORTANTE: Années 1997, 1999, 2001 manquantes")
print("   Les WGI (political_stability, etc.) n'ont été collectés que tous les 2 ans")
print("   de 1996 à 2002. Ceci est NORMAL et reflète la collecte de données WGI.")
print()

# Step 1: Filter to years >= 1996
print("Step 1: Filter to years >= 1996")
df = df[df['Year'] >= 1996].copy()
print(f"After year filter: {len(df)} rows")
print()

# Step 2: Filter to UN recognized countries (using HDI as reference)
print("Step 2: Filter to UN recognized countries")
hdi_df = pd.read_csv(data_dir / "undp_hdi.csv")
valid_countries = hdi_df['Country Name'].unique()
df = df[df['Country Name'].isin(valid_countries)].copy()
print(f"After country filter: {len(df)} rows, {df['Country Name'].nunique()} countries")
print()

# Step 3: DROP gini_index (49% missing - too sparse)
print("Step 3: Drop gini_index (49% missing)")
if 'gini_index' in df.columns:
    missing_gini = df['gini_index'].isnull().sum() / len(df) * 100
    print(f"gini_index missing: {missing_gini:.1f}%")
    df = df.drop(columns=['gini_index'])
    print("✓ Dropped gini_index")
print()

# Define variables
target_col = 'political_stability'
predictor_cols = [
    'gdp_per_capita',
    'gdp_growth',
    'unemployment_ilo',
    'inflation_cpi',
    'trade_gdp_pct',
    'rule_of_law',
    'government_effectiveness',
    'hdi'
]

print(f"Target: {target_col}")
print(f"Predictors ({len(predictor_cols)}):")
for p in predictor_cols:
    print(f"  - {p}")
print()

# Step 4: Missing values summary BEFORE imputation
print("=" * 80)
print("MISSING VALUES BEFORE IMPUTATION")
print("=" * 80)
print()

all_vars = [target_col] + predictor_cols
for var in all_vars:
    missing = df[var].isnull().sum()
    pct = missing / len(df) * 100
    print(f"{var:30s}: {missing:>5} / {len(df)} ({pct:>5.1f}%)")
print()

# Step 5: Impute missing values with linear interpolation
print("=" * 80)
print("IMPUTING MISSING VALUES (Linear Interpolation)")
print("=" * 80)
print()
print("WHY LINEAR INTERPOLATION?")
print("-" * 80)
print("Nous utilisons l'interpolation linéaire au lieu de la moyenne car:")
print("  1. RESPECTE LA STRUCTURE TEMPORELLE")
print("     - Les données économiques évoluent progressivement dans le temps")
print("     - Ex: Si PIB est 35k en 1998 et 38k en 2001, on estime 36k et 37k")
print("     - Une moyenne ignorerait cette tendance temporelle")
print()
print("  2. PRÉSERVE LA VARIANCE")
print("     - La moyenne réduit artificiellement la variance")
print("     - L'interpolation maintient la progression naturelle")
print()
print("  3. ÉVITE LES 'SAUTS' ARTIFICIELS")
print("     - Remplacer par la moyenne crée des discontinuités")
print("     - L'interpolation assure une transition fluide")
print()
print("  4. MEILLEURE POUR L'ANALYSE DE SÉRIES TEMPORELLES")
print("     - Essentiel pour les modèles économétriques (GMM, panel data)")
print("     - Préserve les relations dynamiques entre variables")
print()
print("MÉTHODE APPLIQUÉE:")
print("  - Linear interpolation: pour valeurs manquantes au milieu de la série")
print("  - Forward fill (ffill): pour propager la dernière valeur connue")
print("  - Backward fill (bfill): pour propager depuis la première valeur")
print("=" * 80)
print()

# Sort by country and year to ensure proper time series order
df = df.sort_values(['Country Name', 'Year']).reset_index(drop=True)

for col in all_vars:
    missing_before = df[col].isnull().sum()
    if missing_before > 0:
        # Interpolate with linear method (respects time series structure)
        # Then forward fill and backward fill for edges
        df[col] = df.groupby('Country Name')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
                       .ffill()
                       .bfill()
        )
        missing_after = df[col].isnull().sum()
        imputed = missing_before - missing_after
        print(f"{col:30s}: imputed {imputed:>5} values, {missing_after:>5} still missing")

print()

# Step 6: Drop rows with remaining missing values
print("=" * 80)
print("FINAL CLEANING")
print("=" * 80)
print()

rows_before = len(df)
df_clean = df.dropna(subset=all_vars).copy()
rows_dropped = rows_before - len(df_clean)

print(f"Rows before: {rows_before}")
print(f"Rows dropped (still had missing): {rows_dropped}")
print(f"Rows after: {len(df_clean)}")
print()

# Step 7: Final statistics
print("=" * 80)
print("FINAL DATASET STATISTICS")
print("=" * 80)
print()

print(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"Countries: {df_clean['Country Name'].nunique()}")
print(f"Years: {df_clean['Year'].min():.0f} - {df_clean['Year'].max():.0f}")
print()

# Variables summary
print("Variables (9 total):")
print("  Economic (5):")
print("    - gdp_per_capita")
print("    - gdp_growth")
print("    - unemployment_ilo")
print("    - inflation_cpi")
print("    - trade_gdp_pct")
print("  Governance (2):")
print("    - rule_of_law")
print("    - government_effectiveness")
print("  Development (1):")
print("    - hdi")
print("  Target (1):")
print("    - political_stability")
print()

# Countries with most observations
country_counts = df_clean.groupby('Country Name').size().sort_values(ascending=False)
print("Top 10 countries by observations:")
for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
    print(f"  {i:2d}. {country:30s}: {count:3d} obs")
print()

# Countries with fewest observations
print("Bottom 10 countries by observations:")
for i, (country, count) in enumerate(country_counts.tail(10).items(), 1):
    print(f"  {i:2d}. {country:30s}: {count:3d} obs")
print()

# Step 8: Identify countries with missing data BEFORE imputation
print("=" * 80)
print("COUNTRIES WITH HIGH MISSING DATA (Before Imputation)")
print("=" * 80)
print()

# Recalculate missing for original df (before dropna)
df_orig = pd.read_csv(data_dir / "merged_data.csv")
df_orig = df_orig[df_orig['Year'] >= 1996]
df_orig = df_orig[df_orig['Country Name'].isin(valid_countries)]
df_orig = df_orig.drop(columns=['gini_index'], errors='ignore')

# Calculate missing % per country
country_missing = df_orig.groupby('Country Name').agg({
    col: lambda x: x.isnull().sum() / len(x) * 100
    for col in all_vars
})
country_missing['avg_missing'] = country_missing.mean(axis=1)
country_missing = country_missing.sort_values('avg_missing', ascending=False)

print("Countries with > 20% missing data (before imputation):")
high_missing = country_missing[country_missing['avg_missing'] > 20]
print(f"\nTotal: {len(high_missing)} countries\n")
for country in high_missing.index[:30]:  # Show top 30
    avg_missing = high_missing.loc[country, 'avg_missing']
    n_obs = df_orig[df_orig['Country Name'] == country].shape[0]
    print(f"{country:35s}: {avg_missing:>5.1f}% missing (n={n_obs})")

print()

# Step 9: Save final clean data
print("=" * 80)
print("SAVING FINAL CLEAN DATA")
print("=" * 80)
print()

output_file = data_dir / "final_clean_data.csv"
df_clean.to_csv(output_file, index=False)

print(f"✓ Saved: {output_file.name}")
print(f"  Rows: {len(df_clean)}")
print(f"  Columns: {len(df_clean.columns)}")
print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
print()

print("=" * 80)
print("DONE!")
print("=" * 80)
print()
print("Summary:")
print(f"  - 193 UN countries → {df_clean['Country Name'].nunique()} countries in final dataset")
print(f"  - 9 variables (8 predictors + 1 target)")
print(f"  - {len(df_clean)} complete observations")
print(f"  - Years: {df_clean['Year'].min():.0f}-{df_clean['Year'].max():.0f}")
print(f"  - Ready for ML!")
print()
