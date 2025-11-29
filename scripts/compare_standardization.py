"""
Compare ML Models WITH and WITHOUT Standardization
Run all models and show which performs best
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 100)
print("STANDARDIZATION IMPACT COMPARISON")
print("=" * 100)
print()

# Load data
df = pd.read_csv(data_dir / 'final_clean_data.csv')
print(f"Data loaded: {df.shape[0]} rows, {df['Country Name'].nunique()} countries")
print(f"Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Prepare features and target
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

X = df[predictor_cols].copy()
y = df[target_col].copy()

print(f"Features: {len(predictor_cols)}")
print(f"Target: {target_col}")
print()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} observations")
print(f"Test set: {X_test.shape[0]} observations")
print()

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 100)
print("FEATURE SCALES COMPARISON")
print("=" * 100)
print()
print(f"{'Feature':<30} {'Original Mean':>15} {'Original Std':>15} {'Scaled Mean':>15} {'Scaled Std':>15}")
print("-" * 95)
for col in predictor_cols:
    orig_mean = X_train[col].mean()
    orig_std = X_train[col].std()
    col_idx = predictor_cols.index(col)
    scaled_mean = X_train_scaled[:, col_idx].mean()
    scaled_std = X_train_scaled[:, col_idx].std()
    print(f"{col:<30} {orig_mean:>15.2f} {orig_std:>15.2f} {scaled_mean:>15.2e} {scaled_std:>15.2f}")
print()

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, random_state=42, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF kernel)': SVR(kernel='rbf', C=1.0, gamma='scale')
}

print("=" * 100)
print("TRAINING MODELS")
print("=" * 100)
print()

results = []

for model_name, model in models.items():
    print(f"{model_name}")
    print("-" * 100)

    for standardized in [False, True]:
        label = "WITH standardization" if standardized else "WITHOUT standardization"

        if standardized:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train.values, X_test.values

        # Clone model
        from sklearn.base import clone
        model_clone = clone(model)

        # Fit
        model_clone.fit(X_tr, y_train)

        # Predict
        y_pred_train = model_clone.predict(X_tr)
        y_pred_test = model_clone.predict(X_te)

        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation
        cv_scores = cross_val_score(model_clone, X_tr, y_train, cv=5,
                                     scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            'Model': model_name,
            'Standardized': standardized,
            'R2_train': r2_train,
            'R2_test': r2_test,
            'RMSE_train': rmse_train,
            'RMSE_test': rmse_test,
            'MAE_train': mae_train,
            'MAE_test': mae_test,
            'CV_R2_mean': cv_mean,
            'CV_R2_std': cv_std
        })

        print(f"  {label:30s} | R2 train: {r2_train:.4f} | R2 test: {r2_test:.4f} | "
              f"RMSE test: {rmse_test:.4f} | CV R2: {cv_mean:.4f} (+/- {cv_std:.4f})")

    print()

print()
print("=" * 100)
print("COMPLETE RESULTS TABLE")
print("=" * 100)
print()

results_df = pd.DataFrame(results)
results_df['Standardized_label'] = results_df['Standardized'].map({True: 'WITH', False: 'WITHOUT'})

print(results_df[['Model', 'Standardized_label', 'R2_test', 'RMSE_test', 'MAE_test', 'CV_R2_mean', 'CV_R2_std']].to_string(index=False))
print()

# Calculate impact
print("=" * 100)
print("STANDARDIZATION IMPACT ANALYSIS")
print("=" * 100)
print()

comparison = []

for model_name in models.keys():
    without = results_df[(results_df['Model'] == model_name) & (results_df['Standardized'] == False)].iloc[0]
    with_std = results_df[(results_df['Model'] == model_name) & (results_df['Standardized'] == True)].iloc[0]

    r2_diff = with_std['R2_test'] - without['R2_test']
    rmse_diff = with_std['RMSE_test'] - without['RMSE_test']

    r2_pct_change = (r2_diff / abs(without['R2_test'])) * 100 if without['R2_test'] != 0 else 0
    rmse_pct_change = (rmse_diff / without['RMSE_test']) * 100 if without['RMSE_test'] != 0 else 0

    comparison.append({
        'Model': model_name,
        'R2_without': without['R2_test'],
        'R2_with': with_std['R2_test'],
        'R2_diff': r2_diff,
        'R2_pct_change': r2_pct_change,
        'RMSE_without': without['RMSE_test'],
        'RMSE_with': with_std['RMSE_test'],
        'RMSE_diff': rmse_diff,
        'RMSE_pct_change': rmse_pct_change
    })

comparison_df = pd.DataFrame(comparison)

print("Positive R2_diff = Standardization IMPROVES performance")
print("Negative R2_diff = Standardization HURTS performance")
print()
print(comparison_df.to_string(index=False))
print()

# Key findings
print("=" * 100)
print("KEY FINDINGS")
print("=" * 100)
print()

comparison_df_sorted = comparison_df.sort_values('R2_diff', ascending=False)

print("Models that BENEFIT MOST from standardization (by R2 improvement):")
print("-" * 100)
for idx, row in comparison_df_sorted.head(3).iterrows():
    print(f"  {row['Model']:30s}: R2 improved by {row['R2_diff']:+.4f} ({row['R2_pct_change']:+.2f}%)")
print()

print("Models that BENEFIT LEAST (or hurt) from standardization:")
print("-" * 100)
for idx, row in comparison_df_sorted.tail(3).iterrows():
    print(f"  {row['Model']:30s}: R2 changed by {row['R2_diff']:+.4f} ({row['R2_pct_change']:+.2f}%)")
print()

best_model_without = results_df[results_df['Standardized'] == False].nlargest(1, 'R2_test').iloc[0]
best_model_with = results_df[results_df['Standardized'] == True].nlargest(1, 'R2_test').iloc[0]

print("=" * 100)
print("BEST MODELS")
print("=" * 100)
print()

print("Best performing model WITHOUT standardization:")
print(f"  Model: {best_model_without['Model']}")
print(f"  R2 score: {best_model_without['R2_test']:.4f}")
print(f"  RMSE: {best_model_without['RMSE_test']:.4f}")
print(f"  MAE: {best_model_without['MAE_test']:.4f}")
print(f"  CV R2: {best_model_without['CV_R2_mean']:.4f} (+/- {best_model_without['CV_R2_std']:.4f})")
print()

print("Best performing model WITH standardization:")
print(f"  Model: {best_model_with['Model']}")
print(f"  R2 score: {best_model_with['R2_test']:.4f}")
print(f"  RMSE: {best_model_with['RMSE_test']:.4f}")
print(f"  MAE: {best_model_with['MAE_test']:.4f}")
print(f"  CV R2: {best_model_with['CV_R2_mean']:.4f} (+/- {best_model_with['CV_R2_std']:.4f})")
print()

# Overall winner
all_models_sorted = results_df.sort_values('R2_test', ascending=False)
overall_best = all_models_sorted.iloc[0]

print("=" * 100)
print("OVERALL WINNER")
print("=" * 100)
print()
print(f"Model: {overall_best['Model']}")
print(f"Standardization: {'WITH' if overall_best['Standardized'] else 'WITHOUT'}")
print(f"R2 score: {overall_best['R2_test']:.4f}")
print(f"RMSE: {overall_best['RMSE_test']:.4f}")
print(f"MAE: {overall_best['MAE_test']:.4f}")
print(f"CV R2: {overall_best['CV_R2_mean']:.4f} (+/- {overall_best['CV_R2_std']:.4f})")
print()

print("=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)
print()
print("Models that REQUIRE standardization:")
print("  - SVR (Support Vector Regression)")
print("  - Ridge/Lasso (regularized regression)")
print()
print("Models that DON'T need standardization:")
print("  - Random Forest (tree-based)")
print("  - Gradient Boosting (tree-based)")
print()
print("Models where it's optional:")
print("  - Linear Regression (doesn't affect predictions, but helps numerically)")
print()

# Save results
output_file = data_dir / "standardization_comparison_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file.name}")
print()
