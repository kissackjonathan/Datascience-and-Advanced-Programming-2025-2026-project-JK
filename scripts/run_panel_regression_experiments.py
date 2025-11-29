"""
Run panel regression experiments
Test different Fixed Effects models with FSI and Political Stability targets
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from linearmodels import PanelOLS, RandomEffects, PooledOLS, FirstDifferenceOLS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Paths
processed_dir = Path(__file__).parent.parent / "data" / "processed"
raw_dir = Path(__file__).parent.parent / "data" / "raw"


def load_fsi_panel(csv_path):
    """Load and reshape FSI panel data to long format"""
    df = pd.read_csv(csv_path)

    # Extract year labels from row 1
    years_row = df.iloc[0, 3:].values

    # Remove header rows
    df = df.iloc[1:].reset_index(drop=True)

    # Extract country names
    countries = df.iloc[:, 0]

    # Get indicator names
    indicators = ['Total', 'C1', 'C2', 'C3', 'E1', 'E2', 'E3', 'P1', 'P2', 'P3', 'S1', 'S2', 'X1']

    all_data = []

    for indicator in indicators:
        # Find columns for this indicator
        cols_with_indicator = [i for i, col in enumerate(df.columns[3:]) if indicator in str(col)]

        if len(cols_with_indicator) == 0:
            continue

        # Extract years for these columns
        years = [years_row[i] for i in cols_with_indicator if pd.notna(years_row[i])]

        # Extract data
        data_cols = [3 + i for i in cols_with_indicator[:len(years)]]
        df_indicator = df.iloc[:, [0] + data_cols].copy()
        df_indicator.columns = ['Country'] + [str(int(float(y))) for y in years]

        # Melt to long format
        df_long = df_indicator.melt(id_vars='Country', var_name='Year', value_name=indicator)
        all_data.append(df_long)

    # Merge all indicators - use more memory-efficient approach
    if len(all_data) > 0:
        # Set index before merging for better performance
        for i in range(len(all_data)):
            all_data[i] = all_data[i].set_index(['Country', 'Year'])

        # Concatenate along columns instead of repeated merges
        df_merged = pd.concat(all_data, axis=1)
        df_merged = df_merged.reset_index()

        df_merged['Year'] = pd.to_numeric(df_merged['Year'], errors='coerce')

        # Convert indicators to numeric
        for col in df_merged.columns[2:]:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        df_merged = df_merged.dropna(subset=['Country', 'Year'])
        df_merged = df_merged.dropna(how='all', subset=df_merged.columns[2:])

        return df_merged

    return None


def prepare_panel_data(df, target_col, min_years=5, max_missing_pct=0.5):
    """Clean and prepare panel data"""
    df_clean = df.copy()

    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=[target_col])

    # Get predictor columns
    predictor_cols = [col for col in df_clean.columns if col not in ['Country', 'Year', target_col]]

    print(f"Initial: {len(df_clean)} obs, {df_clean['Country'].nunique()} countries")

    # Calculate missing percentage per country
    country_stats = df_clean.groupby('Country').agg({
        'Year': 'count',
        **{col: lambda x: x.isnull().sum() / len(x) for col in predictor_cols}
    })
    country_stats.columns = ['n_years'] + [f'{col}_missing_pct' for col in predictor_cols]

    # Remove countries with too few years
    countries_enough_years = country_stats[country_stats['n_years'] >= min_years].index

    # Remove countries with too much missing data
    missing_cols = [f'{col}_missing_pct' for col in predictor_cols]
    country_stats['avg_missing'] = country_stats[missing_cols].mean(axis=1)
    countries_low_missing = country_stats[country_stats['avg_missing'] <= max_missing_pct].index

    # Keep only good countries
    valid_countries = list(set(countries_enough_years) & set(countries_low_missing))
    df_clean = df_clean[df_clean['Country'].isin(valid_countries)].copy()

    print(f"After filtering: {len(df_clean)} obs, {len(valid_countries)} countries")

    # Impute missing values with country mean
    for col in predictor_cols:
        missing_before = df_clean[col].isnull().sum()
        if missing_before > 0:
            df_clean[col] = df_clean.groupby('Country')[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Drop any remaining rows with missing values
    df_clean = df_clean.dropna()

    print(f"Final: {len(df_clean)} obs, {df_clean['Country'].nunique()} countries")
    print(f"Years: {df_clean['Year'].min():.0f} - {df_clean['Year'].max():.0f}")

    return df_clean


def evaluate_model(y_true, y_pred):
    """Calculate R², MAE, RMSE"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}


def test_twoway_fe(df_panel, target_col, predictor_cols):
    """Test Two-Way Fixed Effects"""
    y = df_panel[target_col]
    X = df_panel[predictor_cols]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def test_dynamic_panel(df, target_col, predictor_cols):
    """Test Dynamic Panel with lagged dependent variable"""
    df_dynamic = df.copy()

    df_dynamic[f'{target_col}_lag1'] = df_dynamic.groupby(level=0)[target_col].shift(1)
    df_dynamic = df_dynamic.dropna(subset=[f'{target_col}_lag1'])

    y = df_dynamic[target_col]
    X = df_dynamic[predictor_cols + [f'{target_col}_lag1']]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def test_distributed_lags(df, target_col, predictor_cols, n_lags=1):
    """Test model with distributed lags"""
    df_lags = df.copy()

    lagged_cols = []
    for col in predictor_cols:
        for lag in range(1, n_lags + 1):
            lag_col = f'{col}_lag{lag}'
            df_lags[lag_col] = df_lags.groupby(level=0)[col].shift(lag)
            lagged_cols.append(lag_col)

    df_lags = df_lags.dropna()

    y = df_lags[target_col]
    X = df_lags[predictor_cols + lagged_cols]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def test_random_effects(df_panel, target_col, predictor_cols):
    """Test Random Effects model"""
    y = df_panel[target_col]
    X = df_panel[predictor_cols]

    model = RandomEffects(y, X)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def test_pooled_ols(df_panel, target_col, predictor_cols):
    """Test Pooled OLS (no fixed effects)"""
    y = df_panel[target_col]
    X = df_panel[predictor_cols]

    model = PooledOLS(y, X)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def test_first_differences(df, target_col, predictor_cols):
    """Test First Differences model"""
    df_diff = df.copy()

    # First difference all variables
    df_diff[target_col] = df_diff.groupby(level=0)[target_col].diff()
    for col in predictor_cols:
        df_diff[col] = df_diff.groupby(level=0)[col].diff()

    # Drop first observation for each entity (NaN after diff)
    df_diff = df_diff.dropna()

    y = df_diff[target_col]
    X = df_diff[predictor_cols]

    model = PooledOLS(y, X)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    y_pred = results.fitted_values
    metrics = evaluate_model(y, y_pred)

    return results, metrics


def split_train_test(df_panel, test_years=3):
    """Split panel data into train/test by year"""
    # Get the last test_years years for testing
    max_year = df_panel.index.get_level_values('Year').max()
    test_start_year = max_year - test_years + 1

    train = df_panel[df_panel.index.get_level_values('Year') < test_start_year]
    test = df_panel[df_panel.index.get_level_values('Year') >= test_start_year]

    return train, test


def evaluate_on_test(model_results, test_data, target_col, predictor_cols):
    """Evaluate fitted model on test set"""
    try:
        # Get test X and y
        y_test = test_data[target_col]
        X_test = test_data[predictor_cols]

        # Predict (note: this is tricky for panel models with fixed effects)
        # For simplicity, we'll use the model parameters only
        # This won't include entity/time effects
        y_pred = X_test @ model_results.params[predictor_cols]

        metrics = evaluate_model(y_test, y_pred)
        return metrics
    except Exception as e:
        print(f"Warning: Could not evaluate on test set: {e}")
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan}


def main():
    print("=" * 80)
    print("PANEL REGRESSION EXPERIMENTS - POLITICAL STABILITY PREDICTION")
    print("=" * 80)
    print()

    # Load FSI data
    print("Loading FSI data...")
    df_fsi = load_fsi_panel(raw_dir / 'fsi_rankings_2.csv')
    print(f"FSI loaded: {df_fsi.shape}")
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
        'political_stability': 'wb_political_stability.csv'
    }

    wb_data = {}
    for name, filename in wb_files.items():
        df = pd.read_csv(processed_dir / filename)
        df = df.rename(columns={'Country Name': 'Country', name: name})
        df = df[['Country', 'Year', name]]
        wb_data[name] = df

    print("World Bank data loaded")
    print()

    # Prepare FSI target dataset (2006+)
    print("=" * 80)
    print("PREPARING FSI TARGET DATASET (2006+)")
    print("=" * 80)
    df_fsi_target = df_fsi[df_fsi['Year'] >= 2006][['Country', 'Year', 'Total']].copy()
    df_fsi_target = df_fsi_target.rename(columns={'Total': 'target_fsi'})

    # Merge with World Bank
    df_merged_fsi = df_fsi_target.copy()
    for name, df_wb in wb_data.items():
        if name != 'political_stability':
            df_merged_fsi = df_merged_fsi.merge(df_wb, on=['Country', 'Year'], how='left')

    # Clean
    df_fsi_clean = prepare_panel_data(df_merged_fsi, 'target_fsi', min_years=3, max_missing_pct=0.6)

    # USE ALL COUNTRIES (no sampling)
    print(f"Using all {df_fsi_clean['Country'].nunique()} countries with {len(df_fsi_clean)} observations")

    df_fsi_panel = df_fsi_clean.set_index(['Country', 'Year'])
    predictors_fsi = [col for col in df_fsi_panel.columns if col != 'target_fsi']

    print()

    # Prepare Political Stability target dataset (1996+)
    print("=" * 80)
    print("PREPARING POLITICAL STABILITY TARGET DATASET (1996+)")
    print("=" * 80)
    df_polstab = wb_data['political_stability'].copy()
    df_polstab = df_polstab[df_polstab['Year'] >= 1996].copy()
    df_polstab = df_polstab.rename(columns={'political_stability': 'target_polstab'})

    # Merge with World Bank
    df_merged_polstab = df_polstab.copy()
    for name, df_wb in wb_data.items():
        if name != 'political_stability':
            df_merged_polstab = df_merged_polstab.merge(df_wb, on=['Country', 'Year'], how='left')

    # Clean
    df_polstab_clean = prepare_panel_data(df_merged_polstab, 'target_polstab', min_years=5, max_missing_pct=0.6)

    # USE ALL COUNTRIES (no sampling)
    print(f"Using all {df_polstab_clean['Country'].nunique()} countries with {len(df_polstab_clean)} observations")

    df_polstab_panel = df_polstab_clean.set_index(['Country', 'Year'])
    predictors_polstab = [col for col in df_polstab_panel.columns if col != 'target_polstab']

    print()

    # Test models for FSI
    print("=" * 80)
    print("TESTING ALL MODELS FOR TARGET: FSI TOTAL")
    print("=" * 80)
    print()

    results_fsi = {}
    results_fsi_detailed = {}

    print("1. Pooled OLS (baseline - no fixed effects)...")
    model_res, metrics = test_pooled_ols(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['Pooled OLS'] = metrics
    results_fsi_detailed['Pooled OLS'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("2. Two-Way Fixed Effects...")
    model_res, metrics = test_twoway_fe(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['Two-Way FE'] = metrics
    results_fsi_detailed['Two-Way FE'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("3. Random Effects...")
    model_res, metrics = test_random_effects(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['Random Effects'] = metrics
    results_fsi_detailed['Random Effects'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("4. First Differences...")
    model_res, metrics = test_first_differences(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['First Differences'] = metrics
    results_fsi_detailed['First Differences'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("5. Dynamic Panel...")
    model_res, metrics = test_dynamic_panel(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['Dynamic Panel'] = metrics
    results_fsi_detailed['Dynamic Panel'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("6. Distributed Lags...")
    model_res, metrics = test_distributed_lags(df_fsi_panel, 'target_fsi', predictors_fsi)
    results_fsi['Distributed Lags'] = metrics
    results_fsi_detailed['Distributed Lags'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print()

    # Test models for Political Stability
    print("=" * 80)
    print("TESTING ALL MODELS FOR TARGET: POLITICAL STABILITY")
    print("=" * 80)
    print()

    results_polstab = {}
    results_polstab_detailed = {}

    print("1. Pooled OLS (baseline - no fixed effects)...")
    model_res, metrics = test_pooled_ols(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['Pooled OLS'] = metrics
    results_polstab_detailed['Pooled OLS'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("2. Two-Way Fixed Effects...")
    model_res, metrics = test_twoway_fe(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['Two-Way FE'] = metrics
    results_polstab_detailed['Two-Way FE'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("3. Random Effects...")
    model_res, metrics = test_random_effects(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['Random Effects'] = metrics
    results_polstab_detailed['Random Effects'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("4. First Differences...")
    model_res, metrics = test_first_differences(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['First Differences'] = metrics
    results_polstab_detailed['First Differences'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("5. Dynamic Panel...")
    model_res, metrics = test_dynamic_panel(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['Dynamic Panel'] = metrics
    results_polstab_detailed['Dynamic Panel'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print("6. Distributed Lags...")
    model_res, metrics = test_distributed_lags(df_polstab_panel, 'target_polstab', predictors_polstab)
    results_polstab['Distributed Lags'] = metrics
    results_polstab_detailed['Distributed Lags'] = model_res
    print(f"   R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")

    print()

    # Compare results
    print("=" * 80)
    print("MODEL COMPARISON - ALL RESULTS")
    print("=" * 80)
    print()

    comparison_data = []
    for model_name, metrics in results_fsi.items():
        comparison_data.append({
            'Target': 'FSI Total',
            'Model': model_name,
            'R²': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })

    for model_name, metrics in results_polstab.items():
        comparison_data.append({
            'Target': 'Political Stability',
            'Model': model_name,
            'R²': metrics['R2'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('R²', ascending=False)

    print(df_comparison.to_string(index=False))
    print()

    # Find best model
    best_idx = df_comparison['R²'].idxmax()
    best_model = df_comparison.loc[best_idx]

    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Target: {best_model['Target']}")
    print(f"Model: {best_model['Model']}")
    print(f"R²: {best_model['R²']:.4f}")
    print(f"MAE: {best_model['MAE']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")
    print("=" * 80)
    print()

    print("RECOMMENDATION:")
    print(f"Use {best_model['Target']} as target variable")
    print(f"Implement {best_model['Model']} for baseline regression")
    print("=" * 80)
    print()

    # Analyze coefficients of best models
    print("=" * 80)
    print("COEFFICIENT ANALYSIS - TOP MODELS")
    print("=" * 80)
    print()

    # Find best FSI model
    best_fsi_model = max(results_fsi.items(), key=lambda x: x[1]['R2'])
    print(f"Best FSI Model: {best_fsi_model[0]} (R² = {best_fsi_model[1]['R2']:.4f})")
    print("-" * 80)

    if best_fsi_model[0] in results_fsi_detailed:
        model_res = results_fsi_detailed[best_fsi_model[0]]
        print("\nCoefficients:")
        print(model_res.params.to_string())
        print("\nP-values:")
        print(model_res.pvalues.to_string())
        print("\nSignificant predictors (p < 0.05):")
        sig_predictors = model_res.pvalues[model_res.pvalues < 0.05]
        if len(sig_predictors) > 0:
            for pred in sig_predictors.index:
                coef = model_res.params[pred]
                pval = model_res.pvalues[pred]
                print(f"  {pred}: {coef:+.4f} (p={pval:.4f})")
        else:
            print("  No significant predictors at p < 0.05")

    print()
    print("-" * 80)

    # Find best Political Stability model
    best_polstab_model = max(results_polstab.items(), key=lambda x: x[1]['R2'])
    print(f"\nBest Political Stability Model: {best_polstab_model[0]} (R² = {best_polstab_model[1]['R2']:.4f})")
    print("-" * 80)

    if best_polstab_model[0] in results_polstab_detailed:
        model_res = results_polstab_detailed[best_polstab_model[0]]
        print("\nCoefficients:")
        print(model_res.params.to_string())
        print("\nP-values:")
        print(model_res.pvalues.to_string())
        print("\nSignificant predictors (p < 0.05):")
        sig_predictors = model_res.pvalues[model_res.pvalues < 0.05]
        if len(sig_predictors) > 0:
            for pred in sig_predictors.index:
                coef = model_res.params[pred]
                pval = model_res.pvalues[pred]
                print(f"  {pred}: {coef:+.4f} (p={pval:.4f})")
        else:
            print("  No significant predictors at p < 0.05")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
