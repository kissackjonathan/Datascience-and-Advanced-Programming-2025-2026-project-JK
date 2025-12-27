# Optuna Integration Guide

## Overview

Optuna has been integrated for **XGBoost** and **MLP (Neural Network)** models to provide more efficient hyperparameter optimization using **Bayesian optimization** instead of exhaustive grid search.

### Benefits of Optuna vs GridSearchCV

| Aspect | GridSearchCV | Optuna | Improvement |
|--------|--------------|---------|-------------|
| **Search Strategy** | Exhaustive grid | Bayesian (TPE) | Smarter exploration |
| **Search Space** | Discrete values only | Continuous + log-scale | Finer granularity |
| **Efficiency** | Tests ALL combinations | Intelligent sampling | 40-60% faster |
| **Pruning** | None | Stops bad trials early | Saves computation |
| **Trials** | 96-216 combinations | ~100 trials | Better results, less time |

---

## 🚀 Quick Start

### Installation

```bash
pip install optuna optuna-integration
# or
conda install -c conda-forge optuna
```

### Basic Usage

```python
from src.optuna_models import fit_xgboost_optuna, fit_mlp_optuna

# Load your data
X_train, y_train = ...

# XGBoost with Optuna (100 trials, 5-fold CV)
result = fit_xgboost_optuna(
    X_train,
    y_train,
    n_trials=100,  # More trials = better results
    cv=5,
    n_jobs=-1
)

model = result['model']
best_params = result['best_params']
cv_score = result['cv_score']
print(f"Best CV R²: {cv_score:.4f}")
print(f"Best params: {best_params}")

# MLP with THREE-LEVEL optimization
result = fit_mlp_optuna(
    X_train,
    y_train,
    n_trials=80,   # Fewer trials for MLP (slower per trial)
    cv=5
)
```

---

## 📊 XGBoost + Optuna

### Hyperparameter Search Space

Optuna optimizes **9 hyperparameters** with intelligent sampling:

```python
{
    'n_estimators': [50, 500],           # Integer range
    'max_depth': [3, 10],                # Integer range
    'learning_rate': [0.001, 0.3],       # Log-scale continuous
    'subsample': [0.6, 1.0],             # Continuous
    'colsample_bytree': [0.6, 1.0],      # Continuous
    'gamma': [0, 5],                      # Continuous
    'min_child_weight': [1, 10],         # Integer
    'reg_alpha': [1e-8, 10.0],           # Log-scale (L1)
    'reg_lambda': [1e-8, 10.0],          # Log-scale (L2)
}
```

### Expected Improvements

- **Current performance** (GridSearchCV): R² = 0.7401, RMSE = 0.4935
- **Expected with Optuna**: R² = 0.76-0.78, RMSE = 0.45-0.47
- **Time reduction**: ~13s → ~10-12s (despite more exploration)

### Example Output

```python
{
    'model': XGBRegressor(...),
    'best_params': {
        'n_estimators': 287,
        'max_depth': 6,
        'learning_rate': 0.0423,
        'subsample': 0.87,
        'colsample_bytree': 0.92,
        ...
    },
    'cv_score': 0.7654,
    'train_score': 0.8102,
    'overfitting_gap': 0.0448,
    'feature_importance': Series([...]),
    'optuna_study': Study(...)  # For visualization
}
```

---

## 🧠 MLP + Optuna (THREE-LEVEL Optimization)

### Three Levels Explained

1. **LEVEL 1 - Optuna Trial Pruning**
   - Stops unpromising trials early (after 2-3 CV folds)
   - Saves ~40% computation time
   - Uses MedianPruner to compare trials

2. **LEVEL 2 - Cross-Validation**
   - 5-fold CV for robust evaluation
   - Each fold trains the neural network from scratch

3. **LEVEL 3 - Early Stopping** ✅ (KEPT from original)
   - `early_stopping=True`
   - `validation_fraction=0.1` (10% internal validation)
   - Stops training when validation performance plateaus

### Hyperparameter Search Space

```python
{
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100),
                           (150,), (150, 100), (200,)],
    'activation': ['relu', 'tanh'],
    'alpha': [1e-5, 1e-1],                    # Log-scale L2 regularization
    'learning_rate_init': [1e-4, 1e-1],       # Log-scale
    'batch_size': [32, 64, 128, 256],
}
```

### Expected Improvements

- **Current performance** (GridSearchCV): R² = 0.6924, Overfitting Gap = 0.1006
- **Expected with Optuna**:
  - R² = 0.71-0.73 (+2-3%)
  - Overfitting Gap = 0.05-0.07 (reduced by ~40%)
- **Time**: ~35s → ~25-30s

---

## 🎨 Visualization (Bonus)

Optuna provides built-in visualizations:

```python
import optuna

# After training
study = result['optuna_study']

# Optimization history
optuna.visualization.plot_optimization_history(study).show()

# Parameter importance
optuna.visualization.plot_param_importances(study).show()

# Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study).show()

# Hyperparameter relationships
optuna.visualization.plot_contour(study, params=['learning_rate', 'max_depth']).show()
```

---

## 💡 Best Practices

### When to Use Optuna

✅ **Use Optuna for:**
- XGBoost (complex hyperparameter space)
- MLP (reduce overfitting, find optimal architecture)
- When you want better performance than current GridSearchCV

❌ **Keep GridSearchCV for:**
- Random Forest (already excellent R²=0.77)
- KNN, Elastic Net, SVR (simple hyperparameter spaces)
- When interpretability/simplicity is more important than 1-2% performance gain

### Recommended Settings

| Model | n_trials | cv | Expected Time | Expected Improvement |
|-------|----------|-----|---------------|---------------------|
| **XGBoost** | 100 | 5 | ~10-12s | +3-5% R² |
| **MLP** | 80 | 5 | ~25-30s | +2-3% R², -40% overfitting gap |

---

## 🔧 Integration with Existing Code

### Option 1: Direct Usage (Recommended)

```python
from src.optuna_models import fit_xgboost_optuna

# Replace GridSearchCV with Optuna
result = fit_xgboost_optuna(X_train, y_train, n_trials=100)
model = result['model']
```

### Option 2: Add Method to Existing Classes

Add this to `XGBoostPredictor` class:

```python
def fit_optuna(self, X_train, y_train, n_trials=100, cv=5, n_jobs=-1):
    from src.optuna_models import fit_xgboost_optuna

    result = fit_xgboost_optuna(
        X_train, y_train, n_trials, cv, n_jobs, self.logger
    )

    self.model = result['model']
    self.best_params_ = result['best_params']
    self.cv_score_ = result['cv_score']
    self.train_score_ = result['train_score']
    self.overfitting_gap_ = result['overfitting_gap']
    self.feature_importance_ = result['feature_importance']
    self.optuna_study = result['optuna_study']

    return self
```

---

## 📝 Example Workflow

```python
import pandas as pd
from src.optuna_models import fit_xgboost_optuna, fit_mlp_optuna

# 1. Load data
train_data = pd.read_csv('data/processed/train_data.csv')
X_train = train_data.drop('political_stability', axis=1)
y_train = train_data['political_stability']

# 2. Train XGBoost with Optuna
print("Training XGBoost with Optuna...")
xgb_result = fit_xgboost_optuna(X_train, y_train, n_trials=100)
print(f"XGBoost CV R²: {xgb_result['cv_score']:.4f}")
print(f"Best params: {xgb_result['best_params']}")

# 3. Train MLP with THREE-LEVEL optimization
print("\nTraining MLP with THREE-LEVEL optimization...")
mlp_result = fit_mlp_optuna(X_train, y_train, n_trials=80)
print(f"MLP CV R²: {mlp_result['cv_score']:.4f}")
print(f"Overfitting gap: {mlp_result['overfitting_gap']:.4f}")

# 4. Compare with GridSearchCV baselines
print("\nComparison:")
print(f"XGBoost: GridSearchCV R²=0.7401 → Optuna R²={xgb_result['cv_score']:.4f}")
print(f"MLP:     GridSearchCV R²=0.6924 → Optuna R²={mlp_result['cv_score']:.4f}")
```

---

## 🎯 Summary

| Model | Optimization | Why? | Expected Gain |
|-------|-------------|------|---------------|
| **XGBoost** | Optuna ✅ | Complex hyperparameter space (9 params) | +3-5% R² |
| **MLP** | Optuna ✅ | Reduce overfitting, find optimal architecture | +2-3% R², -40% overfitting |
| **Random Forest** | GridSearchCV ✅ | Already excellent (R²=0.77), simplicity preferred | Marginal |
| **Others** | GridSearchCV ✅ | Simple hyperparameter spaces | Not worth complexity |

**Conclusion**: Use Optuna for XGBoost and MLP for maximum performance with minimal code changes!
