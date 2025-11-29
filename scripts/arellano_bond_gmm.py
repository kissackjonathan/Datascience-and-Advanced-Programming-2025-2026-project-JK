"""
Arellano-Bond GMM Implementation
Full implementation with instruments and diagnostic tests
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def create_arellano_bond_instruments(df_panel, target_col, predictor_cols, max_lag=4):
    """
    Create instruments for Arellano-Bond GMM

    Instruments:
    - For Δy_{t-1}: use y_{t-2}, y_{t-3}, ..., y_{t-max_lag}
    - For ΔX_t: use X_t (assumed exogenous, can instrument with themselves)

    Returns first-differenced data with instruments
    """
    print(f"Creating Arellano-Bond instruments (max lag = {max_lag})...")

    df = df_panel.copy()

    # Create lags of target
    for lag in range(1, max_lag + 1):
        df[f'{target_col}_lag{lag}'] = df.groupby(level=0)[target_col].shift(lag)

    # Create lags of predictors (for instruments)
    for col in predictor_cols:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df.groupby(level=0)[col].shift(lag)

    # First difference all variables
    df_diff = df.copy()
    df_diff[f'd_{target_col}'] = df_diff.groupby(level=0)[target_col].diff()
    df_diff[f'd_{target_col}_lag1'] = df_diff.groupby(level=0)[f'{target_col}_lag1'].diff()

    for col in predictor_cols:
        df_diff[f'd_{col}'] = df_diff.groupby(level=0)[col].diff()

    # Drop rows with NaN (first few obs per entity)
    df_diff = df_diff.dropna()

    print(f"  Observations after differencing and creating lags: {len(df_diff)}")

    return df_diff


def arellano_bond_gmm(df_panel, target_col, predictor_cols, max_lag=4):
    """
    Arellano-Bond GMM Estimator

    Model: Δy_it = ρ Δy_{i,t-1} + β' ΔX_it + Δu_it

    Instruments for Δy_{i,t-1}: y_{i,t-2}, y_{i,t-3}, ..., y_{i,t-max_lag}
    Instruments for ΔX_it: X_it (assumed exogenous)
    """
    print("\n" + "=" * 70)
    print("ARELLANO-BOND GMM ESTIMATION")
    print("=" * 70)

    # Create instruments
    df_diff = create_arellano_bond_instruments(df_panel, target_col, predictor_cols, max_lag)

    # Dependent variable: Δy_t
    y = df_diff[f'd_{target_col}'].values

    # Endogenous regressors: Δy_{t-1}, ΔX_t
    endog_vars = [f'd_{target_col}_lag1'] + [f'd_{col}' for col in predictor_cols]
    X_endog = df_diff[endog_vars].values

    # Instruments: levels of y (t-2, t-3, ...) and levels of X (t-1, t-2, ...)
    instrument_vars = []

    # Instruments for Δy_{t-1}: y_{t-2}, y_{t-3}, ...
    for lag in range(2, max_lag + 1):
        if f'{target_col}_lag{lag}' in df_diff.columns:
            instrument_vars.append(f'{target_col}_lag{lag}')

    # Instruments for ΔX: use levels X_{t-1}
    for col in predictor_cols:
        if f'{col}_lag1' in df_diff.columns:
            instrument_vars.append(f'{col}_lag1')

    Z = df_diff[instrument_vars].values

    print(f"\nModel specification:")
    print(f"  Dependent: Δ{target_col}")
    print(f"  Endogenous: {endog_vars}")
    print(f"  Instruments ({len(instrument_vars)}): {instrument_vars[:10]}...")
    print(f"  Observations: {len(y)}")

    # Two-Stage GMM estimation
    # Stage 1: Use identity weighting matrix
    print("\nStage 1: Initial GMM estimation...")

    # Project X onto Z: X_hat = Z(Z'Z)^{-1}Z'X
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    P_Z = Z @ ZtZ_inv @ Z.T
    X_hat = P_Z @ X_endog

    # First stage regression: y = X_hat β + error
    beta_1 = np.linalg.inv(X_hat.T @ X_hat) @ (X_hat.T @ y)
    resid_1 = y - X_endog @ beta_1

    print(f"  Stage 1 residual std: {resid_1.std():.4f}")

    # Stage 2: Optimal weighting matrix
    print("Stage 2: Optimal GMM with robust weighting...")

    # Moment conditions: Z'u (should be zero)
    moments = Z.T @ resid_1.reshape(-1, 1)

    # Robust weighting matrix: W = (Z'uu'Z)^{-1}
    # Omega = (1/n) Σ Z_i u_i^2 Z_i'
    u_squared = resid_1**2
    Omega = (Z.T * u_squared) @ Z  # (p x n) * (n,) @ (n x p) = (p x p)
    W = np.linalg.inv(Omega)

    # Efficient GMM: min (Z'u)' W (Z'u)
    # FOC: X'Z W Z'X β = X'Z W Z'y
    ZtX = Z.T @ X_endog
    Zty = Z.T @ y

    beta_gmm = np.linalg.inv(ZtX.T @ W @ ZtX) @ (ZtX.T @ W @ Zty)

    # Final residuals
    resid_gmm = y - X_endog @ beta_gmm

    print(f"  GMM residual std: {resid_gmm.std():.4f}")

    # Standard errors (robust)
    n = len(y)
    k = len(beta_gmm)

    # Robust variance: (X'Z W Z'X)^{-1}
    var_beta = np.linalg.inv(ZtX.T @ W @ ZtX) / n
    se_beta = np.sqrt(np.diag(var_beta))

    # T-statistics
    t_stats = beta_gmm / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

    # Results dataframe
    results_df = pd.DataFrame({
        'Variable': endog_vars,
        'Coefficient': beta_gmm,
        'Std.Error': se_beta,
        't-stat': t_stats,
        'P>|t|': p_values
    })

    print("\n" + "=" * 70)
    print("COEFFICIENT ESTIMATES")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Diagnostics
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TESTS")
    print("=" * 70)

    # 1. Sargan test of overidentifying restrictions
    # H0: instruments are valid (moments = 0)
    # Test statistic: n * (Z'u)' W (Z'u) ~ χ²(#instruments - #parameters)

    moments_final = Z.T @ resid_gmm.reshape(-1, 1)
    sargan_stat = n * (moments_final.T @ W @ moments_final)[0, 0]
    sargan_df = len(instrument_vars) - len(endog_vars)
    sargan_pval = 1 - stats.chi2.cdf(sargan_stat, sargan_df)

    print(f"\n1. Sargan Test (Over-identification)")
    print(f"   H0: Instruments are valid")
    print(f"   Statistic: {sargan_stat:.4f}")
    print(f"   DOF: {sargan_df}")
    print(f"   P-value: {sargan_pval:.4f}")
    if sargan_pval > 0.05:
        print(f"   ✓ PASS: Cannot reject H0 (instruments appear valid)")
    else:
        print(f"   ✗ FAIL: Reject H0 (instruments may be invalid)")

    # 2. AR(1) test - autocorrelation in first-differenced errors
    # Expected: negative correlation (by construction)

    # Get residuals by entity
    df_diff['residual'] = resid_gmm

    ar1_correlations = []
    for entity in df_diff.index.get_level_values(0).unique():
        entity_data = df_diff.loc[entity]
        if len(entity_data) > 1:
            resid_entity = entity_data['residual'].values
            # Autocorrelation
            if len(resid_entity) > 1:
                resid_t = resid_entity[1:]
                resid_t_minus_1 = resid_entity[:-1]
                if len(resid_t) > 0:
                    ar1_correlations.append(np.corrcoef(resid_t, resid_t_minus_1)[0, 1])

    if len(ar1_correlations) > 0:
        ar1_corr = np.nanmean(ar1_correlations)
        # Test if significantly different from zero
        ar1_stat = ar1_corr * np.sqrt(len(ar1_correlations))
        ar1_pval = 2 * (1 - stats.norm.cdf(np.abs(ar1_stat)))

        print(f"\n2. AR(1) Test (First-order autocorrelation in differences)")
        print(f"   H0: No autocorrelation")
        print(f"   Mean correlation: {ar1_corr:.4f}")
        print(f"   Test statistic: {ar1_stat:.4f}")
        print(f"   P-value: {ar1_pval:.4f}")
        if ar1_pval < 0.05:
            print(f"   ✓ EXPECTED: Reject H0 (autocorrelation present, as expected from differencing)")
        else:
            print(f"   ⚠ UNEXPECTED: Cannot reject H0")
    else:
        print(f"\n2. AR(1) Test: Not enough data")

    # 3. AR(2) test - second-order autocorrelation
    # Should NOT be present (would invalidate instruments)

    ar2_correlations = []
    for entity in df_diff.index.get_level_values(0).unique():
        entity_data = df_diff.loc[entity]
        if len(entity_data) > 2:
            resid_entity = entity_data['residual'].values
            if len(resid_entity) > 2:
                resid_t = resid_entity[2:]
                resid_t_minus_2 = resid_entity[:-2]
                if len(resid_t) > 0:
                    ar2_correlations.append(np.corrcoef(resid_t, resid_t_minus_2)[0, 1])

    if len(ar2_correlations) > 0:
        ar2_corr = np.nanmean(ar2_correlations)
        ar2_stat = ar2_corr * np.sqrt(len(ar2_correlations))
        ar2_pval = 2 * (1 - stats.norm.cdf(np.abs(ar2_stat)))

        print(f"\n3. AR(2) Test (Second-order autocorrelation in differences)")
        print(f"   H0: No second-order autocorrelation")
        print(f"   Mean correlation: {ar2_corr:.4f}")
        print(f"   Test statistic: {ar2_stat:.4f}")
        print(f"   P-value: {ar2_pval:.4f}")
        if ar2_pval > 0.05:
            print(f"   ✓ PASS: Cannot reject H0 (no AR(2), instruments are valid)")
        else:
            print(f"   ✗ FAIL: Reject H0 (AR(2) present, instruments may be invalid)")
    else:
        print(f"\n3. AR(2) Test: Not enough data")

    # Performance metrics (on differenced data)
    mae = mean_absolute_error(y, X_endog @ beta_gmm)
    ss_res = np.sum(resid_gmm**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    print("\n" + "=" * 70)
    print("MODEL FIT (on first-differenced data)")
    print("=" * 70)
    print(f"R² = {r2:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {np.sqrt(ss_res/n):.4f}")

    return {
        'coefficients': beta_gmm,
        'std_errors': se_beta,
        't_stats': t_stats,
        'p_values': p_values,
        'variable_names': endog_vars,
        'residuals': resid_gmm,
        'sargan_stat': sargan_stat,
        'sargan_pval': sargan_pval,
        'r2': r2,
        'mae': mae,
        'df_diff': df_diff
    }


def main():
    print("=" * 70)
    print("ARELLANO-BOND GMM - FULL IMPLEMENTATION")
    print("=" * 70)
    print()

    # Load data (same as test_arellano_bond.py)
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    print("Loading World Bank data...")
    wb_files = {
        'gdp_per_capita': 'wb_gdp_per_capita.csv',
        'gdp_growth': 'wb_gdp_growth.csv',
        'unemployment_ilo': 'wb_unemployment_ilo.csv',
        'inflation_cpi': 'wb_inflation_cpi.csv',
        'gini_index': 'wb_gini_index.csv',
        'trade_gdp_pct': 'wb_trade_gdp_pct.csv',
        'political_stability': 'wb_political_stability.csv'
    }

    wb_data = {}
    for name, filename in wb_files.items():
        df_temp = pd.read_csv(data_dir / filename)
        df_temp = df_temp.rename(columns={'Country Name': 'Country', name: name})
        df_temp = df_temp[['Country', 'Year', name]]
        wb_data[name] = df_temp

    # Merge data
    df = wb_data['political_stability'].copy()
    df = df[df['Year'] >= 1996].copy()

    for name, df_wb in wb_data.items():
        if name != 'political_stability':
            df = df.merge(df_wb, on=['Country', 'Year'], how='left')

    # Clean data (same logic as before)
    print("\nCleaning data...")
    target_col = 'political_stability'
    predictor_cols = [
        'gdp_per_capita',
        'gdp_growth',
        'unemployment_ilo',
        'inflation_cpi',
        'gini_index',
        'trade_gdp_pct'
    ]

    df = df.dropna(subset=[target_col])

    # Filter countries
    country_stats = df.groupby('Country').agg({
        'Year': 'count',
        **{col: lambda x: x.isnull().sum() / len(x) for col in predictor_cols}
    })
    country_stats.columns = ['n_years'] + [f'{col}_missing_pct' for col in predictor_cols]

    countries_enough_years = country_stats[country_stats['n_years'] >= 5].index
    missing_cols = [f'{col}_missing_pct' for col in predictor_cols]
    country_stats['avg_missing'] = country_stats[missing_cols].mean(axis=1)
    countries_low_missing = country_stats[country_stats['avg_missing'] <= 0.6].index

    valid_countries = list(set(countries_enough_years) & set(countries_low_missing))
    df = df[df['Country'].isin(valid_countries)].copy()

    # Impute missing
    for col in predictor_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

    df = df.dropna()

    print(f"Final: {len(df)} obs, {df['Country'].nunique()} countries")
    print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")

    # Set up panel structure
    df_panel = df.copy()
    df_panel['country_id'] = pd.Categorical(df_panel['Country']).codes
    df_panel = df_panel.set_index(['country_id', 'Year'])
    df_panel = df_panel.sort_index()
    df_panel = df_panel[[target_col] + predictor_cols]

    print(f"\nPanel: {len(df_panel)} obs, {df_panel.index.get_level_values(0).nunique()} entities")

    # Run Arellano-Bond GMM
    results = arellano_bond_gmm(df_panel, target_col, predictor_cols, max_lag=4)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Arellano-Bond GMM successfully estimated!

Key Results:
- Lag coefficient (ρ): {results['coefficients'][0]:.4f} (p={results['p_values'][0]:.4f})
- R² (on differences): {results['r2']:.4f}
- MAE (on differences): {results['mae']:.4f}

Diagnostic Tests:
- Sargan test p-value: {results['sargan_pval']:.4f} {'✓' if results['sargan_pval'] > 0.05 else '✗'}

Interpretation:
- This is the econometrically rigorous approach
- Corrects for Nickell bias via GMM with instruments
- Valid for causal inference (if tests pass)
""")


if __name__ == "__main__":
    main()
