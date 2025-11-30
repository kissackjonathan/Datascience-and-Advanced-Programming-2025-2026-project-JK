"""
Dynamic Panel Data Analysis with Fixed Effects
Execute panel regression and display results
"""
import warnings

import numpy as np
import pandas as pd
from linearmodels import PanelOLS, RandomEffects
from linearmodels.panel import compare
from pathlib import Path

warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 100)
print("DYNAMIC PANEL DATA ANALYSIS - POLITICAL STABILITY")
print("=" * 100)
print()

# Load data
df = pd.read_csv(data_dir / 'final_clean_data.csv')

print(f"Dataset: {len(df):,} observations, {df['Country Name'].nunique()} countries")
print(f"Period: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Variables
target = 'political_stability'
predictors = [
    'gdp_per_capita',
    'gdp_growth',
    'unemployment_ilo',
    'inflation_cpi',
    'trade_gdp_pct',
    'rule_of_law',
    'government_effectiveness',
    'hdi'
]

# Panel structure
df_panel = df.set_index(['Country Name', 'Year']).sort_index()

print("=" * 100)
print("1. FIXED EFFECTS (FE) MODEL")
print("=" * 100)
print()

y = df_panel[target]
X = df_panel[predictors]

# Fixed Effects with entity and time effects
fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True)
fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

print(fe_results.summary)
print()

# Extract key coefficients
fe_coefs = pd.DataFrame({
    'Coefficient': fe_results.params,
    'Std_Error': fe_results.std_errors,
    'p_value': fe_results.pvalues
})
fe_coefs['Significant'] = fe_coefs['p_value'] < 0.05

print("\n" + "=" * 100)
print("COEFFICIENT SUMMARY")
print("=" * 100)
print()
print(f"{'Variable':<30} {'Coefficient':>12} {'Std Error':>12} {'p-value':>10} {'Significant':>12}")
print("-" * 100)
for var, row in fe_coefs.iterrows():
    sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
    print(f"{var:<30} {row['Coefficient']:>12.4f} {row['Std_Error']:>12.4f} {row['p_value']:>10.4f} {sig:>12}")
print()
print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
print()

# Model fit
print("=" * 100)
print("MODEL FIT STATISTICS")
print("=" * 100)
print()
print(f"R-squared (within):    {fe_results.rsquared_within:.4f}")
print(f"R-squared (between):   {fe_results.rsquared_between:.4f}")
print(f"R-squared (overall):   {fe_results.rsquared_overall:.4f}")
print(f"F-statistic:           {fe_results.f_statistic.stat:.2f}")
print(f"F-statistic p-value:   {fe_results.f_statistic.pval:.6f}")
print(f"Number of observations: {fe_results.nobs:,.0f}")
print(f"Number of entities:    {fe_results.entity_info.total}")
print(f"Number of time periods: {fe_results.time_info.total}")
print()

# Dynamic panel
print("=" * 100)
print("2. DYNAMIC PANEL MODEL (with lagged dependent variable)")
print("=" * 100)
print()

# Create lag
df_panel_sorted = df_panel.sort_index()
df_panel_sorted['political_stability_lag1'] = df_panel_sorted.groupby(level=0)[target].shift(1)
df_dynamic = df_panel_sorted.dropna(subset=['political_stability_lag1'])

print(f"Dynamic panel observations: {len(df_dynamic):,}")
print(f"Observations lost due to lagging: {len(df_panel) - len(df_dynamic):,}")
print()

y_dynamic = df_dynamic[target]
X_dynamic = df_dynamic[predictors + ['political_stability_lag1']]

# Dynamic FE
dynamic_fe = PanelOLS(y_dynamic, X_dynamic, entity_effects=True, time_effects=True)
dynamic_fe_results = dynamic_fe.fit(cov_type='clustered', cluster_entity=True)

print(dynamic_fe_results.summary)
print()

# Extract dynamic coefficients
dynamic_coefs = pd.DataFrame({
    'Coefficient': dynamic_fe_results.params,
    'Std_Error': dynamic_fe_results.std_errors,
    'p_value': dynamic_fe_results.pvalues
})

print("\n" + "=" * 100)
print("DYNAMIC PANEL COEFFICIENTS")
print("=" * 100)
print()
print(f"{'Variable':<30} {'Coefficient':>12} {'Std Error':>12} {'p-value':>10} {'Significant':>12}")
print("-" * 100)
for var, row in dynamic_coefs.iterrows():
    sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
    print(f"{var:<30} {row['Coefficient']:>12.4f} {row['Std_Error']:>12.4f} {row['p_value']:>10.4f} {sig:>12}")
print()

# Persistence analysis
lag_coef = dynamic_coefs.loc['political_stability_lag1', 'Coefficient']
lag_pval = dynamic_coefs.loc['political_stability_lag1', 'p_value']

print("=" * 100)
print("PERSISTENCE ANALYSIS")
print("=" * 100)
print()
print(f"Lagged coefficient (ρ): {lag_coef:.4f}")
print(f"P-value: {lag_pval:.6f}")
print(f"Significant: {'YES ***' if lag_pval < 0.01 else 'YES **' if lag_pval < 0.05 else 'NO'}")
print()

if 0 < lag_coef < 1:
    half_life = -np.log(2) / np.log(lag_coef)
    print(f"Half-life of shocks: {half_life:.2f} years")
    print(f"  → A shock to political stability decays by 50% in {half_life:.1f} years")
    print()

    if lag_coef > 0.7:
        print("Interpretation: STRONG persistence - stability is highly inertial")
    elif lag_coef > 0.4:
        print("Interpretation: MODERATE persistence - some adjustment over time")
    else:
        print("Interpretation: WEAK persistence - rapid adjustment to shocks")
else:
    print("Interpretation: Model suggests instability or explosive dynamics")
print()

# Key findings
print("=" * 100)
print("KEY FINDINGS")
print("=" * 100)
print()

sig_vars_static = fe_coefs[fe_coefs['Significant']].sort_values('Coefficient', ascending=False)
sig_vars_dynamic = dynamic_coefs[dynamic_coefs['Significant']].sort_values('Coefficient', ascending=False)

print("1. MOST IMPORTANT PREDICTORS (Static FE):")
print("-" * 100)
for idx, (var, row) in enumerate(sig_vars_static.head(3).iterrows(), 1):
    print(f"   {idx}. {var:30s}: β = {row['Coefficient']:+.4f} (p = {row['p_value']:.4f})")
print()

print("2. MOST IMPORTANT PREDICTORS (Dynamic FE):")
print("-" * 100)
for idx, (var, row) in enumerate(sig_vars_dynamic.head(3).iterrows(), 1):
    print(f"   {idx}. {var:30s}: β = {row['Coefficient']:+.4f} (p = {row['p_value']:.4f})")
print()

print("3. MODEL COMPARISON:")
print("-" * 100)
print(f"   Static FE R² (within):  {fe_results.rsquared_within:.4f}")
print(f"   Dynamic FE R² (within): {dynamic_fe_results.rsquared_within:.4f}")
print(f"   Improvement: {(dynamic_fe_results.rsquared_within - fe_results.rsquared_within)*100:+.2f}%")
print()

print("4. INTERPRETATION:")
print("-" * 100)
print("   The dynamic panel model reveals:")
print(f"   - Political stability shows {'strong' if lag_coef > 0.7 else 'moderate' if lag_coef > 0.4 else 'weak'} persistence (ρ={lag_coef:.3f})")
print("   - Country fixed effects capture unobserved heterogeneity")
print("   - Time fixed effects control for global shocks")
if len(sig_vars_static) > 0:
    top_var = sig_vars_static.index[0]
    top_coef = sig_vars_static.loc[top_var, 'Coefficient']
    print(f"   - {top_var} is the strongest predictor (β={top_coef:+.4f})")
print()

print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
