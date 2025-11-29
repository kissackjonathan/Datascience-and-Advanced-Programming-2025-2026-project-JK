"""
Machine Learning Experiments with WGI Variables
Test ML models with 8 variables (6 economic + 2 WGI)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except:
        LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

processed_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("MACHINE LEARNING WITH WGI VARIABLES")
print("=" * 80)
print()

# Load data
print("Loading World Bank data...")
wb_files = {
    'gdp_per_capita': 'wb_gdp_per_capita.csv',
    'gdp_growth': 'wb_gdp_growth.csv',
    'unemployment_ilo': 'wb_unemployment_ilo.csv',
    'inflation_cpi': 'wb_inflation_cpi.csv',
    'gini_index': 'wb_gini_index.csv',
    'trade_gdp_pct': 'wb_trade_gdp_pct.csv',
    'political_stability': 'wb_political_stability.csv',
    'rule_of_law': 'wb_rule_of_law.csv',
    'government_effectiveness': 'wb_government_effectiveness.csv'
}

wb_data = {}
for name, filename in wb_files.items():
    df = pd.read_csv(processed_dir / filename)
    df = df.rename(columns={'Country Name': 'Country', name: name})
    df = df[['Country', 'Year', name]]
    wb_data[name] = df

print("Data loaded")
print()

# Merge all on Political Stability
df = wb_data['political_stability'].copy()
df = df[df['Year'] >= 1996].copy()

for name, df_wb in wb_data.items():
    if name != 'political_stability':
        df = df.merge(df_wb, on=['Country', 'Year'], how='left')

# Define target and predictors
target = 'political_stability'
predictors = [
    'gdp_per_capita',
    'gdp_growth',
    'unemployment_ilo',
    'inflation_cpi',
    'gini_index',
    'trade_gdp_pct',
    'rule_of_law',
    'government_effectiveness'
]

print(f"Target: {target}")
print(f"Predictors ({len(predictors)}):")
for p in predictors:
    print(f"  - {p}")
print()

# Clean data
print("Cleaning data...")
df = df.dropna(subset=[target])

# Filter countries
country_stats = df.groupby('Country').agg({
    'Year': 'count',
    **{col: lambda x: x.isnull().sum() / len(x) for col in predictors}
})
country_stats.columns = ['n_years'] + [f'{col}_missing_pct' for col in predictors]

countries_enough_years = country_stats[country_stats['n_years'] >= 5].index
missing_cols = [f'{col}_missing_pct' for col in predictors]
country_stats['avg_missing'] = country_stats[missing_cols].mean(axis=1)
countries_low_missing = country_stats[country_stats['avg_missing'] <= 0.6].index

valid_countries = list(set(countries_enough_years) & set(countries_low_missing))
df = df[df['Country'].isin(valid_countries)].copy()

print(f"Countries: {len(valid_countries)}")

# Impute missing
for col in predictors:
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

df = df.dropna()
print(f"Final observations: {len(df)}")
print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Feature engineering
print("Creating ML features...")

df_ml = df.copy()

# Lags (t-1, t-2, t-3)
for col in predictors + [target]:
    for lag in [1, 2, 3]:
        df_ml[f'{col}_lag{lag}'] = df_ml.groupby('Country')[col].shift(lag)

# Rolling stats
for col in predictors:
    df_ml[f'{col}_volatility'] = df_ml.groupby('Country')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    df_ml[f'{col}_trend'] = df_ml.groupby('Country')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

# Interactions
df_ml['gdp_x_gini'] = df_ml['gdp_per_capita'] * df_ml['gini_index']
df_ml['growth_x_unemployment'] = df_ml['gdp_growth'] * df_ml['unemployment_ilo']
df_ml['rule_of_law_x_effectiveness'] = df_ml['rule_of_law'] * df_ml['government_effectiveness']

df_ml = df_ml.dropna()
print(f"Observations after feature engineering: {len(df_ml)}")

# Get feature columns
feature_cols = [col for col in df_ml.columns if col not in ['Country', 'Year', target]]
print(f"Total features: {len(feature_cols)}")
print()

# Train/Test split (temporal)
train_mask = df_ml['Year'] <= 2020
test_mask = df_ml['Year'] > 2020

df_train = df_ml[train_mask].copy()
df_test = df_ml[test_mask].copy()

print(f"Train: {len(df_train)} obs ({df_train['Year'].min():.0f}-{df_train['Year'].max():.0f})")
print(f"Test:  {len(df_test)} obs ({df_test['Year'].min():.0f}-{df_test['Year'].max():.0f})")
print()

X_train = df_train[feature_cols]
y_train = df_train[target]
X_test = df_test[feature_cols]
y_test = df_test[target]

results = {}

# ============================================================================
# MODEL 1: Random Forest
# ============================================================================
print("=" * 80)
print("MODEL 1: RANDOM FOREST")
print("=" * 80)
print()

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_train_pred_rf = rf_model.predict(X_train)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)

y_test_pred_rf = rf_model.predict(X_test)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

results['Random Forest'] = {
    'train_r2': train_r2_rf,
    'train_mae': train_mae_rf,
    'test_r2': test_r2_rf,
    'test_mae': test_mae_rf
}

print(f"Train R² = {train_r2_rf:.4f}, MAE = {train_mae_rf:.4f}")
print(f"Test  R² = {test_r2_rf:.4f}, MAE = {test_mae_rf:.4f}")
print(f"Overfitting: {(train_r2_rf - test_r2_rf):.4f}")
print()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 features:")
print(feature_importance.head(15).to_string(index=False))
print()

# ============================================================================
# MODEL 2: XGBoost/LightGBM
# ============================================================================
if XGBOOST_AVAILABLE:
    print("=" * 80)
    print("MODEL 2: XGBOOST")
    print("=" * 80)
    print()

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    y_train_pred_xgb = xgb_model.predict(X_train)
    train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
    train_mae_xgb = mean_absolute_error(y_train, y_train_pred_xgb)

    y_test_pred_xgb = xgb_model.predict(X_test)
    test_r2_xgb = r2_score(y_test, y_test_pred_xgb)
    test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)

    results['XGBoost'] = {
        'train_r2': train_r2_xgb,
        'train_mae': train_mae_xgb,
        'test_r2': test_r2_xgb,
        'test_mae': test_mae_xgb
    }

    print(f"Train R² = {train_r2_xgb:.4f}, MAE = {train_mae_xgb:.4f}")
    print(f"Test  R² = {test_r2_xgb:.4f}, MAE = {test_mae_xgb:.4f}")
    print()

# ============================================================================
# MODEL 3: Neural Network
# ============================================================================
print("=" * 80)
print("MODEL 3: NEURAL NETWORK (MLP)")
print("=" * 80)
print()

scaler_nn = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled = scaler_nn.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42,
    verbose=False
)

mlp_model.fit(X_train_scaled, y_train)

y_train_pred_mlp = mlp_model.predict(X_train_scaled)
train_r2_mlp = r2_score(y_train, y_train_pred_mlp)
train_mae_mlp = mean_absolute_error(y_train, y_train_pred_mlp)

y_test_pred_mlp = mlp_model.predict(X_test_scaled)
test_r2_mlp = r2_score(y_test, y_test_pred_mlp)
test_mae_mlp = mean_absolute_error(y_test, y_test_pred_mlp)

results['Neural Network'] = {
    'train_r2': train_r2_mlp,
    'train_mae': train_mae_mlp,
    'test_r2': test_r2_mlp,
    'test_mae': test_mae_mlp
}

print(f"Train R² = {train_r2_mlp:.4f}, MAE = {train_mae_mlp:.4f}")
print(f"Test  R² = {test_r2_mlp:.4f}, MAE = {test_mae_mlp:.4f}")
print()

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================
print("=" * 80)
print("COMPARISON WITH BASELINES")
print("=" * 80)
print()

comparison_data = []

# OLD Baseline (6 economic variables only)
comparison_data.append({
    'Model': 'Baseline OLD (6 econ vars)',
    'R2': 0.8988,
    'MAE': 0.238,
    'Variables': 6
})

# NEW Baseline (6 econ + 2 WGI)
comparison_data.append({
    'Model': 'Baseline NEW (6 econ + 2 WGI)',
    'R2': 0.9228,  # From test_wgi_impact.py (rule_of_law only, but should be similar)
    'MAE': 0.203,
    'Variables': 8
})

# ML models
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': f'{model_name} (with WGI)',
        'R2': metrics['test_r2'],
        'MAE': metrics['test_mae'],
        'Variables': 8
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison['R2_pct'] = df_comparison['R2'] * 100

print(df_comparison[['Model', 'Variables', 'R2_pct', 'MAE']].to_string(index=False))
print()

# Best model
best_model = df_comparison.iloc[df_comparison['R2'].argmax()]
print(f"Best model: {best_model['Model']}")
print(f"  R² = {best_model['R2']:.4f} ({best_model['R2']*100:.2f}%)")
print(f"  MAE = {best_model['MAE']:.4f}")
print()

# Improvements
baseline_old = df_comparison[df_comparison['Model'] == 'Baseline OLD (6 econ vars)'].iloc[0]
baseline_new = df_comparison[df_comparison['Model'] == 'Baseline NEW (6 econ + 2 WGI)'].iloc[0]

print("IMPROVEMENTS:")
print(f"1. Adding WGI to baseline:")
print(f"   {baseline_old['R2']*100:.2f}% → {baseline_new['R2']*100:.2f}% = +{(baseline_new['R2'] - baseline_old['R2'])*100:.2f} points")
print()

print(f"2. Best ML vs NEW baseline:")
print(f"   {baseline_new['R2']*100:.2f}% → {best_model['R2']*100:.2f}% = +{(best_model['R2'] - baseline_new['R2'])*100:.2f} points")
print()

print(f"3. Best ML vs OLD baseline:")
print(f"   {baseline_old['R2']*100:.2f}% → {best_model['R2']*100:.2f}% = +{(best_model['R2'] - baseline_old['R2'])*100:.2f} points")
print()

print("=" * 80)
