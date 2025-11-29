"""
Test Impact of WGI Variables on Dynamic Panel Model

Compares:
1. Baseline: 6 economic variables (R² = 89.70%)
2. + WGI variables: 9 variables total
3. Individual WGI impact analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("TESTING WGI VARIABLES IMPACT ON DYNAMIC PANEL MODEL")
print("=" * 80)
print()

# Load World Bank data
print("Loading World Bank data...")
wb_files = {
    'gdp_per_capita': 'wb_gdp_per_capita.csv',
    'gdp_growth': 'wb_gdp_growth.csv',
    'unemployment_ilo': 'wb_unemployment_ilo.csv',
    'inflation_cpi': 'wb_inflation_cpi.csv',
    'gini_index': 'wb_gini_index.csv',
    'trade_gdp_pct': 'wb_trade_gdp_pct.csv',
    'political_stability': 'wb_political_stability.csv',
    # NEW: WGI variables
    'rule_of_law': 'wb_rule_of_law.csv',
    'control_of_corruption': 'wb_control_of_corruption.csv',
    'government_effectiveness': 'wb_government_effectiveness.csv'
}

wb_data = {}
for name, filename in wb_files.items():
    df_temp = pd.read_csv(data_dir / filename)
    df_temp = df_temp.rename(columns={'Country Name': 'Country', name: name})
    df_temp = df_temp[['Country', 'Year', name]]
    wb_data[name] = df_temp

print("Data loaded")
print()

# Merge all data on Political Stability
df = wb_data['political_stability'].copy()
df = df[df['Year'] >= 1996].copy()

for name, df_wb in wb_data.items():
    if name != 'political_stability':
        df = df.merge(df_wb, on=['Country', 'Year'], how='left')

# Define target and predictors
target_col = 'political_stability'
baseline_predictors = [
    'gdp_per_capita',
    'gdp_growth',
    'unemployment_ilo',
    'inflation_cpi',
    'gini_index',
    'trade_gdp_pct'
]
wgi_predictors = [
    'rule_of_law',
    'control_of_corruption',
    'government_effectiveness'
]

# Clean data
print("Cleaning data...")
print(f"Initial: {len(df)} obs, {df['Country'].nunique()} countries")

# Remove rows with missing target
df = df.dropna(subset=[target_col])

# Filter countries: min 5 years, max 60% missing on baseline predictors
all_predictors = baseline_predictors + wgi_predictors

country_stats = df.groupby('Country').agg({
    'Year': 'count',
    **{col: lambda x: x.isnull().sum() / len(x) for col in all_predictors}
})
country_stats.columns = ['n_years'] + [f'{col}_missing_pct' for col in all_predictors]

# Keep countries with >= 5 years
countries_enough_years = country_stats[country_stats['n_years'] >= 5].index

# Keep countries with <= 60% missing on average (for baseline predictors only)
missing_cols_baseline = [f'{col}_missing_pct' for col in baseline_predictors]
country_stats['avg_missing_baseline'] = country_stats[missing_cols_baseline].mean(axis=1)
countries_low_missing = country_stats[country_stats['avg_missing_baseline'] <= 0.6].index

# Filter to valid countries
valid_countries = list(set(countries_enough_years) & set(countries_low_missing))
df = df[df['Country'].isin(valid_countries)].copy()

print(f"After filtering: {len(df)} obs, {len(valid_countries)} countries")

# Impute missing values with country mean (for all predictors)
for col in all_predictors:
    missing_before = df[col].isnull().sum()
    if missing_before > 0:
        df[col] = df.groupby('Country')[col].transform(
            lambda x: x.fillna(x.mean())
        )
        print(f"  Imputed {missing_before} missing values in {col}")

# Drop any remaining rows with missing
df = df.dropna()

print(f"Final: {len(df)} obs, {df['Country'].nunique()} countries")
print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Set up panel structure
df_panel = df.copy()
df_panel['country_id'] = pd.Categorical(df_panel['Country']).codes
df_panel = df_panel.set_index(['country_id', 'Year'])
df_panel = df_panel.sort_index()

# Keep only target and predictors
df_panel = df_panel[[target_col] + all_predictors]

print(f"Panel data:")
print(f"  Observations: {len(df_panel)}")
print(f"  Countries: {df_panel.index.get_level_values(0).nunique()}")
print(f"  Time periods: {df_panel.index.get_level_values(1).nunique()}")
print()

# Function to test dynamic panel
def test_dynamic_panel(df_panel, target_col, predictor_cols, model_name=""):
    """Test Dynamic Panel (FE + Lag) with given predictors"""
    df_dynamic = df_panel.copy()
    df_dynamic[f'{target_col}_lag1'] = df_dynamic.groupby(level=0)[target_col].shift(1)
    df_dynamic = df_dynamic.dropna()

    X = df_dynamic[predictor_cols + [f'{target_col}_lag1']]
    y = df_dynamic[target_col]

    # Two-way FE (entity + time effects)
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    # Metrics
    y_pred = results.predict(fitted=True)
    r2_within = results.rsquared
    r2_overall = results.rsquared_overall
    mae = mean_absolute_error(y, y_pred)

    return {
        'model_name': model_name,
        'n_predictors': len(predictor_cols),
        'R2_within': r2_within,
        'R2_overall': r2_overall,
        'MAE': mae,
        'results': results,
        'n_obs': len(y)
    }

# ============================================================================
# TEST 1: BASELINE (6 economic variables)
# ============================================================================
print("=" * 80)
print("TEST 1: BASELINE (6 ECONOMIC VARIABLES)")
print("=" * 80)
print()

baseline = test_dynamic_panel(df_panel, target_col, baseline_predictors, "Baseline (Economic)")

print(f"Model: {baseline['model_name']}")
print(f"  N predictors: {baseline['n_predictors']}")
print(f"  N observations: {baseline['n_obs']}")
print(f"  R² (Within): {baseline['R2_within']:.4f} ({baseline['R2_within']*100:.2f}%)")
print(f"  R² (Overall): {baseline['R2_overall']:.4f} ({baseline['R2_overall']*100:.2f}%)")
print(f"  MAE: {baseline['MAE']:.4f}")
print()

# ============================================================================
# TEST 2: FULL MODEL (6 economic + 3 WGI variables)
# ============================================================================
print("=" * 80)
print("TEST 2: FULL MODEL (6 ECONOMIC + 3 WGI VARIABLES)")
print("=" * 80)
print()

full_predictors = baseline_predictors + wgi_predictors
full = test_dynamic_panel(df_panel, target_col, full_predictors, "Full (Economic + WGI)")

print(f"Model: {full['model_name']}")
print(f"  N predictors: {full['n_predictors']}")
print(f"  N observations: {full['n_obs']}")
print(f"  R² (Within): {full['R2_within']:.4f} ({full['R2_within']*100:.2f}%)")
print(f"  R² (Overall): {full['R2_overall']:.4f} ({full['R2_overall']*100:.2f}%)")
print(f"  MAE: {full['MAE']:.4f}")
print()

# ============================================================================
# TEST 3: INDIVIDUAL WGI VARIABLES
# ============================================================================
print("=" * 80)
print("TEST 3: INDIVIDUAL WGI IMPACT")
print("=" * 80)
print()

individual_results = {}
for wgi_var in wgi_predictors:
    predictors = baseline_predictors + [wgi_var]
    result = test_dynamic_panel(df_panel, target_col, predictors, f"Baseline + {wgi_var}")
    individual_results[wgi_var] = result

    print(f"Model: {result['model_name']}")
    print(f"  R² (Overall): {result['R2_overall']:.4f} ({result['R2_overall']*100:.2f}%)")
    print(f"  MAE: {result['MAE']:.4f}")
    print(f"  Δ R² vs baseline: {(result['R2_overall'] - baseline['R2_overall'])*100:+.2f} points")
    print()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()

summary_data = []

# Baseline
summary_data.append({
    'Model': 'Baseline (6 economic vars)',
    'N_Vars': baseline['n_predictors'],
    'R2_Overall': baseline['R2_overall'],
    'MAE': baseline['MAE'],
    'Delta_R2': 0.0
})

# Individual WGI
for wgi_var in wgi_predictors:
    result = individual_results[wgi_var]
    summary_data.append({
        'Model': f'+ {wgi_var}',
        'N_Vars': result['n_predictors'],
        'R2_Overall': result['R2_overall'],
        'MAE': result['MAE'],
        'Delta_R2': result['R2_overall'] - baseline['R2_overall']
    })

# Full model
summary_data.append({
    'Model': 'Full (6 econ + 3 WGI)',
    'N_Vars': full['n_predictors'],
    'R2_Overall': full['R2_overall'],
    'MAE': full['MAE'],
    'Delta_R2': full['R2_overall'] - baseline['R2_overall']
})

df_summary = pd.DataFrame(summary_data)
df_summary['R2_Overall_pct'] = df_summary['R2_Overall'] * 100
df_summary['Delta_R2_pct'] = df_summary['Delta_R2'] * 100

print(df_summary[['Model', 'N_Vars', 'R2_Overall_pct', 'MAE', 'Delta_R2_pct']].to_string(index=False))
print()

# ============================================================================
# COEFFICIENT ANALYSIS (FULL MODEL)
# ============================================================================
print("=" * 80)
print("COEFFICIENT ANALYSIS - FULL MODEL")
print("=" * 80)
print()

print("Coefficients:")
print(full['results'].params.to_string())
print()

print("P-values:")
print(full['results'].pvalues.to_string())
print()

print("Significant variables (p < 0.05):")
sig_vars = full['results'].pvalues[full['results'].pvalues < 0.05]
if len(sig_vars) > 0:
    for var in sig_vars.index:
        coef = full['results'].params[var]
        pval = full['results'].pvalues[var]
        print(f"  {var:30s}: coef={coef:+.4f}, p={pval:.4f}")
else:
    print("  No significant variables at p < 0.05")
print()

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print()

best_improvement = max(summary_data[1:-1], key=lambda x: x['Delta_R2'])

print(f"1. Adding ALL 3 WGI variables:")
print(f"   - R² improvement: {full['Delta_R2']*100:+.2f} points")
print(f"   - From {baseline['R2_overall']*100:.2f}% → {full['R2_overall']*100:.2f}%")
print(f"   - MAE improvement: {baseline['MAE'] - full['MAE']:+.4f}")
print()

print(f"2. Best individual WGI variable: {best_improvement['Model']}")
print(f"   - R² improvement: {best_improvement['Delta_R2']*100:+.2f} points")
print()

print(f"3. Significant WGI variables in full model:")
for wgi_var in wgi_predictors:
    if wgi_var in sig_vars.index:
        print(f"   ✓ {wgi_var}: significant (p={full['results'].pvalues[wgi_var]:.4f})")
    else:
        print(f"   ✗ {wgi_var}: not significant (p={full['results'].pvalues[wgi_var]:.4f})")
print()

print("=" * 80)
