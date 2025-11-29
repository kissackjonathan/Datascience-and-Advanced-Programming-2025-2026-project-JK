"""
Machine Learning Experiments for Political Stability Prediction
Test Random Forest, XGBoost, Semi-Supervised, and Unsupervised methods vs Panel Regression baseline
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"WARNING: XGBoost not available: {e}")
    print("Will try LightGBM as alternative...")
    XGBOOST_AVAILABLE = False
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
        print("LightGBM available, will use as XGBoost alternative")
    except Exception:
        LIGHTGBM_AVAILABLE = False
        print("LightGBM also not available")
    print()

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

    # Get indicator names
    indicators = ['Total']

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

    # Merge all indicators
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


def create_ml_features(df, predictors, target_col):
    """
    Create features for ML models including:
    - Lag features
    - Volatility features
    - Trend features
    - Interactions
    """
    df_ml = df.copy()

    # Sort by country and year
    df_ml = df_ml.sort_values(['Country', 'Year'])

    print("Creating ML features...")

    # 1. Lag features (1 and 2 years)
    for col in [target_col] + predictors:
        df_ml[f'{col}_lag1'] = df_ml.groupby('Country')[col].shift(1)
        df_ml[f'{col}_lag2'] = df_ml.groupby('Country')[col].shift(2)

    # 2. Volatility features (rolling std over last 3 years)
    for col in predictors:
        df_ml[f'{col}_volatility'] = df_ml.groupby('Country')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )

    # 3. Trend features (difference from previous year)
    for col in predictors:
        df_ml[f'{col}_trend'] = df_ml.groupby('Country')[col].diff()

    # 4. Interaction features (key interactions)
    df_ml['gdp_x_gini'] = df_ml['gdp_per_capita'] * df_ml['gini_index']
    df_ml['gdp_growth_x_inflation'] = df_ml['gdp_growth'] * df_ml['inflation_cpi']

    # Drop rows with NaN (from lag/rolling operations)
    df_ml = df_ml.dropna()

    print(f"Features created: {len(df_ml.columns)} total columns")
    print(f"Remaining observations: {len(df_ml)}")

    return df_ml


def add_unsupervised_features(df, predictors, n_clusters=5, n_pca=3):
    """
    Add unsupervised learning features:
    - K-Means clustering assignments
    - PCA components
    - Distance to cluster centroids
    """
    print(f"Adding unsupervised features (K-Means with {n_clusters} clusters, PCA with {n_pca} components)...")

    df_unsup = df.copy()

    # Select predictor columns for clustering
    X = df[predictors].values

    # Standardize features for clustering and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_unsup['cluster'] = kmeans.fit_predict(X_scaled)

    # Distance to cluster center (instability measure)
    distances = kmeans.transform(X_scaled)
    df_unsup['distance_to_center'] = distances.min(axis=1)

    # 2. PCA for dimensionality reduction
    pca = PCA(n_components=n_pca, random_state=42)
    pca_components = pca.fit_transform(X_scaled)

    for i in range(n_pca):
        df_unsup[f'pca_{i+1}'] = pca_components[:, i]

    print(f"  Added {n_clusters + n_pca + 1} unsupervised features")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return df_unsup, kmeans, pca, scaler


def pseudo_label_regressor(X_labeled, y_labeled, X_unlabeled, base_estimator, confidence_threshold=0.3, max_iter=3):
    """
    Custom Semi-Supervised Learning using Pseudo-Labeling

    Algorithm:
    1. Train model on labeled data
    2. Predict on unlabeled data
    3. Select high-confidence predictions (low residual variance)
    4. Add pseudo-labeled samples to training set
    5. Retrain and repeat

    Args:
        confidence_threshold: Proportion of unlabeled data to add per iteration (e.g., 0.3 = top 30%)
    """
    import copy

    # Initialize
    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_unlabeled_pool = X_unlabeled.copy()

    for iteration in range(max_iter):
        # Train on current labeled data
        model = copy.deepcopy(base_estimator)
        model.fit(X_train, y_train)

        if len(X_unlabeled_pool) == 0:
            break

        # Predict on unlabeled data
        y_pred_unlabeled = model.predict(X_unlabeled_pool)

        # For Random Forest, use prediction variance as confidence measure
        if hasattr(model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X_unlabeled_pool) for tree in model.estimators_])
            prediction_std = tree_predictions.std(axis=0)

            # Select samples with low variance (high confidence)
            n_to_label = int(confidence_threshold * len(X_unlabeled_pool))
            confident_idx = np.argsort(prediction_std)[:n_to_label]
        else:
            # For other models, just take a random subset
            n_to_label = int(confidence_threshold * len(X_unlabeled_pool))
            confident_idx = np.random.choice(len(X_unlabeled_pool), n_to_label, replace=False)

        # Add pseudo-labeled samples to training set
        if isinstance(X_train, pd.DataFrame):
            X_newly_labeled = X_unlabeled_pool.iloc[confident_idx]
            X_train = pd.concat([X_train, X_newly_labeled], axis=0)
            y_newly_labeled = pd.Series(y_pred_unlabeled[confident_idx], index=X_newly_labeled.index)
            y_train = pd.concat([y_train, y_newly_labeled], axis=0)
            X_unlabeled_pool = X_unlabeled_pool.drop(X_newly_labeled.index)
        else:
            X_newly_labeled = X_unlabeled_pool[confident_idx]
            X_train = np.vstack([X_train, X_newly_labeled])
            y_train = np.concatenate([y_train, y_pred_unlabeled[confident_idx]])
            X_unlabeled_pool = np.delete(X_unlabeled_pool, confident_idx, axis=0)

    # Final training on all data
    final_model = copy.deepcopy(base_estimator)
    final_model.fit(X_train, y_train)

    return final_model


def evaluate_model(y_true, y_pred):
    """Calculate R², MAE, RMSE"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}


def main():
    print("=" * 80)
    print("MACHINE LEARNING EXPERIMENTS - POLITICAL STABILITY PREDICTION")
    print("=" * 80)
    print()

    # Load World Bank data (including Political Stability)
    print("Loading World Bank data...")
    wb_files = {
        'gdp_per_capita': 'wb_gdp_per_capita.csv',
        'gdp_growth': 'wb_gdp_growth.csv',
        'unemployment_ilo': 'wb_unemployment_ilo.csv',
        'inflation_cpi': 'wb_inflation_cpi.csv',
        'gini_index': 'wb_gini_index.csv',
        'trade_gdp_pct': 'wb_trade_gdp_pct.csv',
        'political_stability': 'wb_political_stability.csv',
    }

    wb_data = {}
    for name, filename in wb_files.items():
        df = pd.read_csv(processed_dir / filename)
        df = df.rename(columns={'Country Name': 'Country', name: name})
        df = df[['Country', 'Year', name]]
        wb_data[name] = df

    print("World Bank data loaded")
    print()

    # Prepare Political Stability target dataset (1996+)
    print("=" * 80)
    print("PREPARING DATA - POLITICAL STABILITY TARGET")
    print("=" * 80)
    df_polstab_target = wb_data['political_stability'].copy()
    df_polstab_target = df_polstab_target.rename(columns={'political_stability': 'target_polstab'})
    df_polstab_target = df_polstab_target[df_polstab_target['Year'] >= 1996]  # Use from 1996+

    # Merge with other World Bank indicators
    df_merged_polstab = df_polstab_target.copy()
    for name, df_wb in wb_data.items():
        if name != 'political_stability':
            df_merged_polstab = df_merged_polstab.merge(df_wb, on=['Country', 'Year'], how='left')

    # Clean
    df_polstab_clean = prepare_panel_data(df_merged_polstab, 'target_polstab', min_years=5, max_missing_pct=0.6)

    print(f"Political Stability data: {len(df_polstab_clean)} obs, {df_polstab_clean['Country'].nunique()} countries")
    print(f"Years: {df_polstab_clean['Year'].min():.0f} - {df_polstab_clean['Year'].max():.0f}")

    print()

    # Base predictors
    predictors = ['gdp_per_capita', 'gdp_growth', 'unemployment_ilo',
                  'inflation_cpi', 'gini_index', 'trade_gdp_pct']

    # Create ML features
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    df_ml = create_ml_features(df_polstab_clean, predictors, 'target_polstab')

    # Add unsupervised features
    df_ml, kmeans_model, pca_model, scaler_model = add_unsupervised_features(df_ml, predictors, n_clusters=5, n_pca=3)

    # Get feature names (exclude Country, Year, target)
    feature_cols = [col for col in df_ml.columns if col not in ['Country', 'Year', 'target_polstab']]

    print(f"Using {len(feature_cols)} features:")
    print(f"  Original predictors: {len(predictors)}")
    print(f"  Engineered features (lags, volatility, trends, interactions): {len(feature_cols) - len(predictors) - 9}")
    print(f"  Unsupervised features (clustering, PCA): 9")
    print()

    # Train/Test Split (temporal)
    print("=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)

    train_mask = df_ml['Year'] <= 2020
    test_mask = df_ml['Year'] > 2020

    df_train = df_ml[train_mask].copy()
    df_test = df_ml[test_mask].copy()

    print(f"Training set: {len(df_train)} obs ({df_train['Year'].min():.0f}-{df_train['Year'].max():.0f})")
    print(f"Test set:     {len(df_test)} obs ({df_test['Year'].min():.0f}-{df_test['Year'].max():.0f})")
    print()

    # Prepare X and y
    X_train = df_train[feature_cols]
    y_train = df_train['target_polstab']
    X_test = df_test[feature_cols]
    y_test = df_test['target_polstab']

    # Store results
    results = {}

    # ========================================================================
    # MODEL 1: Random Forest
    # ========================================================================
    print("=" * 80)
    print("MODEL 1: RANDOM FOREST")
    print("=" * 80)
    print()

    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Train predictions
    y_train_pred_rf = rf.predict(X_train)
    metrics_train_rf = evaluate_model(y_train, y_train_pred_rf)

    # Test predictions
    y_test_pred_rf = rf.predict(X_test)
    metrics_test_rf = evaluate_model(y_test, y_test_pred_rf)

    results['Random Forest'] = {
        'train': metrics_train_rf,
        'test': metrics_test_rf,
        'model': rf
    }

    print(f"Train R² = {metrics_train_rf['R2']:.4f}, MAE = {metrics_train_rf['MAE']:.4f}, RMSE = {metrics_train_rf['RMSE']:.4f}")
    print(f"Test  R² = {metrics_test_rf['R2']:.4f}, MAE = {metrics_test_rf['MAE']:.4f}, RMSE = {metrics_test_rf['RMSE']:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print()
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    print()

    # ========================================================================
    # MODEL 2: XGBoost
    # ========================================================================
    if XGBOOST_AVAILABLE:
        print("=" * 80)
        print("MODEL 2: XGBOOST")
        print("=" * 80)
        print()

        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)

        # Train predictions
        y_train_pred_xgb = xgb_model.predict(X_train)
        metrics_train_xgb = evaluate_model(y_train, y_train_pred_xgb)

        # Test predictions
        y_test_pred_xgb = xgb_model.predict(X_test)
        metrics_test_xgb = evaluate_model(y_test, y_test_pred_xgb)

        results['XGBoost'] = {
            'train': metrics_train_xgb,
            'test': metrics_test_xgb,
            'model': xgb_model
        }

        print(f"Train R² = {metrics_train_xgb['R2']:.4f}, MAE = {metrics_train_xgb['MAE']:.4f}, RMSE = {metrics_train_xgb['RMSE']:.4f}")
        print(f"Test  R² = {metrics_test_xgb['R2']:.4f}, MAE = {metrics_test_xgb['MAE']:.4f}, RMSE = {metrics_test_xgb['RMSE']:.4f}")

        # Feature importance
        feature_importance_xgb = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print()
        print("Top 10 Most Important Features:")
        print(feature_importance_xgb.head(10).to_string(index=False))
        print()
    elif LIGHTGBM_AVAILABLE:
        print("=" * 80)
        print("MODEL 2: LIGHTGBM (XGBoost alternative)")
        print("=" * 80)
        print()

        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        lgb_model.fit(X_train, y_train)

        # Train predictions
        y_train_pred_lgb = lgb_model.predict(X_train)
        metrics_train_lgb = evaluate_model(y_train, y_train_pred_lgb)

        # Test predictions
        y_test_pred_lgb = lgb_model.predict(X_test)
        metrics_test_lgb = evaluate_model(y_test, y_test_pred_lgb)

        results['LightGBM'] = {
            'train': metrics_train_lgb,
            'test': metrics_test_lgb,
            'model': lgb_model
        }

        print(f"Train R² = {metrics_train_lgb['R2']:.4f}, MAE = {metrics_train_lgb['MAE']:.4f}, RMSE = {metrics_train_lgb['RMSE']:.4f}")
        print(f"Test  R² = {metrics_test_lgb['R2']:.4f}, MAE = {metrics_test_lgb['MAE']:.4f}, RMSE = {metrics_test_lgb['RMSE']:.4f}")

        # Feature importance
        feature_importance_lgb = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print()
        print("Top 10 Most Important Features:")
        print(feature_importance_lgb.head(10).to_string(index=False))
        print()
    else:
        print("=" * 80)
        print("MODEL 2: XGBOOST/LIGHTGBM - SKIPPED (not available)")
        print("=" * 80)
        print()

    # ========================================================================
    # MODEL 3: SEMI-SUPERVISED (Pseudo-Labeling)
    # ========================================================================
    print("=" * 80)
    print("MODEL 3: SEMI-SUPERVISED (Pseudo-Labeling)")
    print("=" * 80)
    print()

    print("Training Pseudo-Labeling Regressor (Semi-Supervised)...")
    print("Simulating unlabeled data by masking 30% of training labels...")

    # Create semi-supervised scenario: split training into labeled and unlabeled
    n_unlabeled = int(0.3 * len(X_train))
    unlabeled_idx = np.random.RandomState(42).choice(len(X_train), n_unlabeled, replace=False)
    labeled_mask = np.ones(len(X_train), dtype=bool)
    labeled_mask[unlabeled_idx] = False

    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]
    X_unlabeled = X_train[~labeled_mask]

    # Base estimator for semi-supervised learning
    base_estimator = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    # Train with pseudo-labeling
    print(f"Initial labeled: {len(X_labeled)}, unlabeled: {len(X_unlabeled)}")
    semi_supervised_model = pseudo_label_regressor(
        X_labeled, y_labeled, X_unlabeled,
        base_estimator=base_estimator,
        confidence_threshold=0.3,
        max_iter=3
    )

    # Train predictions
    y_train_pred_semi = semi_supervised_model.predict(X_train)
    metrics_train_semi = evaluate_model(y_train, y_train_pred_semi)

    # Test predictions
    y_test_pred_semi = semi_supervised_model.predict(X_test)
    metrics_test_semi = evaluate_model(y_test, y_test_pred_semi)

    results['Pseudo-Labeling (Semi-Supervised)'] = {
        'train': metrics_train_semi,
        'test': metrics_test_semi,
        'model': semi_supervised_model
    }

    print(f"Train R² = {metrics_train_semi['R2']:.4f}, MAE = {metrics_train_semi['MAE']:.4f}, RMSE = {metrics_train_semi['RMSE']:.4f}")
    print(f"Test  R² = {metrics_test_semi['R2']:.4f}, MAE = {metrics_test_semi['MAE']:.4f}, RMSE = {metrics_test_semi['RMSE']:.4f}")
    print(f"Strategy: Start with 70% labeled, iteratively add high-confidence pseudo-labels")
    print()

    # ========================================================================
    # MODEL 4: NEURAL NETWORK (MLP)
    # ========================================================================
    print("=" * 80)
    print("MODEL 4: NEURAL NETWORK (Multi-Layer Perceptron)")
    print("=" * 80)
    print()

    print("Training Neural Network (MLP)...")
    print("Architecture: 3 hidden layers (100, 50, 25 neurons)")
    print("Activation: ReLU, Solver: Adam, Early stopping enabled")

    # Standardize features for neural network
    scaler_nn = StandardScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_test_scaled = scaler_nn.transform(X_test)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
        activation='relu',                  # ReLU activation
        solver='adam',                      # Adam optimizer
        alpha=0.001,                        # L2 regularization
        learning_rate='adaptive',           # Adaptive learning rate
        learning_rate_init=0.001,          # Initial learning rate
        max_iter=500,                       # Max iterations
        early_stopping=True,               # Early stopping to prevent overfitting
        validation_fraction=0.1,           # 10% for validation
        n_iter_no_change=20,               # Stop if no improvement for 20 iterations
        random_state=42,
        verbose=False
    )

    mlp_model.fit(X_train_scaled, y_train)

    # Train predictions
    y_train_pred_mlp = mlp_model.predict(X_train_scaled)
    metrics_train_mlp = evaluate_model(y_train, y_train_pred_mlp)

    # Test predictions
    y_test_pred_mlp = mlp_model.predict(X_test_scaled)
    metrics_test_mlp = evaluate_model(y_test, y_test_pred_mlp)

    results['Neural Network (MLP)'] = {
        'train': metrics_train_mlp,
        'test': metrics_test_mlp,
        'model': mlp_model
    }

    print(f"Train R² = {metrics_train_mlp['R2']:.4f}, MAE = {metrics_train_mlp['MAE']:.4f}, RMSE = {metrics_train_mlp['RMSE']:.4f}")
    print(f"Test  R² = {metrics_test_mlp['R2']:.4f}, MAE = {metrics_test_mlp['MAE']:.4f}, RMSE = {metrics_test_mlp['RMSE']:.4f}")
    print(f"Convergence: Stopped at iteration {mlp_model.n_iter_} (loss={mlp_model.loss_:.4f})")
    print()

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()

    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Train_R2': model_results['train']['R2'],
            'Train_MAE': model_results['train']['MAE'],
            'Test_R2': model_results['test']['R2'],
            'Test_MAE': model_results['test']['MAE'],
            'Overfitting': model_results['train']['R2'] - model_results['test']['R2']
        })

    # Add baseline (from panel regression)
    comparison_data.append({
        'Model': 'Dynamic Panel (baseline)',
        'Train_R2': 0.8494,
        'Train_MAE': 8.1984,
        'Test_R2': 0.4596,
        'Test_MAE': 14.6059,
        'Overfitting': 0.3898
    })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test_R2', ascending=False)

    print(df_comparison.to_string(index=False))
    print()

    # Find best model
    best_model = df_comparison.iloc[0]

    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Model: {best_model['Model']}")
    print(f"Test R²: {best_model['Test_R2']:.4f}")
    print(f"Test MAE: {best_model['Test_MAE']:.4f}")
    print(f"Overfitting: {best_model['Overfitting']:.4f}")
    print("=" * 80)
    print()

    # Analysis
    print("=" * 80)
    print("INSIGHTS")
    print("=" * 80)
    print()

    if best_model['Model'] in ['Random Forest', 'XGBoost', 'LightGBM', 'Pseudo-Labeling (Semi-Supervised)', 'Neural Network (MLP)']:
        print("✓ ML models outperform panel regression on test set!")
        print()
        print("Why ML is better:")
        print("  1. Captures non-linear relationships")
        print("  2. Handles interactions automatically")
        print("  3. Robust to outliers")
        print("  4. Uses engineered features (lags, volatility, trends)")
        print("  5. Leverages unsupervised features (K-Means clustering, PCA)")
        if best_model['Model'] == 'Pseudo-Labeling (Semi-Supervised)':
            print("  6. Semi-supervised approach effectively uses unlabeled data via pseudo-labeling")
        elif best_model['Model'] == 'Neural Network (MLP)':
            print("  6. Deep learning with multiple layers captures complex patterns")
        print()
    else:
        print("Panel regression still best on test set")
        print("ML models may need more tuning")

    print("Overfitting comparison:")
    for _, row in df_comparison.iterrows():
        overfitting_level = "Low" if row['Overfitting'] < 0.1 else "Medium" if row['Overfitting'] < 0.2 else "High"
        print(f"  {row['Model']:30s}: {row['Overfitting']:.4f} ({overfitting_level})")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print(f"Use {best_model['Model']} for prediction")
    print(f"Expected test MAE: {best_model['Test_MAE']:.2f} points")
    print(f"Expected test R²: {best_model['Test_R2']:.4f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
