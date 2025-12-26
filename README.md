# Political Stability Prediction

**Author:** Jonathan Kissack
**Course:** Data Science and Advanced Programming 2025-2026

---

## Vue d'ensemble

Ce projet utilise des techniques de **Machine Learning** et d'**Econométrie de panel** pour prédire la stabilité politique des pays à partir d'indicateurs économiques et sociaux. Le système compare 7 algorithmes ML avec un modèle Dynamic Panel (Fixed Effects) sur des données de 186 pays de 2000 à 2023.

**Objectif:** Prédire la stabilité politique future (2018-2023) en utilisant uniquement des données passées (2000-2017), permettant d'évaluer la capacité prédictive réelle des modèles.

---

## Architecture du Projet

```
Datascience-and-Advanced-Programming-2025-2026-project-JK/
├── main/
│   ├── main.py           # Point d'entrée principal (menu interactif)
│   └── dashboard.py      # Dashboard Streamlit pour visualisation
├── src/
│   ├── data_loader.py    # Chargement et nettoyage des données
│   ├── models.py         # 7 modèles ML + Dynamic Panel
│   └── evaluation.py     # Visualisations et métriques
├── data/
│   ├── raw/              # Données brutes (World Bank, UNDP)
│   └── processed/        # Données traitées (vide jusqu'à exécution)
├── results/              # Résultats et visualisations (vide jusqu'à exécution)
├── models/               # Modèles sauvegardés
└── tests/
    └── test.py           # 67 tests unitaires
```

---

## Comment ça marche

### 1. Pipeline de Données

#### Chargement (`src/data_loader.py`)
- **Sources:**
  - World Bank API (PIB, chômage, inflation, commerce)
  - UNDP (Human Development Index)
  - Worldwide Governance Indicators (stabilité politique, état de droit, efficacité)

#### Nettoyage
- Filtrage par liste blanche UN (186 pays)
- Suppression des outliers (IQR method)
- Gestion des valeurs manquantes
- Standardisation des features

#### Split Temporel
```
Train: 2000-2017 (18 ans, ~2,790 observations)
Test:  2018-2023 (6 ans, 1,116 observations)
```
**Crucial:** Le split est temporel (pas aléatoire) pour simuler une vraie prédiction du futur.

---

### 2. Modèles de Machine Learning

Le système entraîne **7 modèles ML** avec optimisation automatique des hyperparamètres (GridSearchCV + 5-fold CV):

| Modèle | Description | Temps d'entraînement |
|--------|-------------|----------------------|
| **Random Forest** | Ensemble de decision trees | ~103s |
| **XGBoost** | Gradient boosting optimisé | ~15s |
| **Gradient Boosting** | Boosting classique | ~60s |
| **Elastic Net** | Régression L1+L2 | ~5s |
| **SVR** | Support Vector Regression | ~8s |
| **KNN** | K-Nearest Neighbors | ~2s |
| **MLP** | Neural Network (Multi-Layer Perceptron) | ~30s |

#### Optimisation des Hyperparamètres
Chaque modèle teste automatiquement des centaines de combinaisons:
- **Random Forest:** 216 combinaisons (n_estimators, max_depth, min_samples, etc.)
- **XGBoost:** 192 combinaisons (learning_rate, max_depth, regularization, etc.)
- **Elastic Net:** 100 combinaisons (alpha, l1_ratio)

#### Métriques d'Évaluation
- **R² Score:** % de variance expliquée (0.76 = 76% pour Random Forest)
- **MAE (Mean Absolute Error):** Erreur moyenne absolue
- **RMSE:** Erreur quadratique moyenne
- **Overfitting Gap:** Train R² - CV R² (détecte le surapprentissage)

---

### 3. Dynamic Panel Analysis (Fixed Effects)

Modèle économétrique de panel avec:
- **Fixed Effects:** Contrôle des différences entre pays
- **Lagged Dependent Variable:** Stabilité politique passée comme prédicteur
- **Clustered Standard Errors:** Gestion de l'autocorrélation et hétéroscédasticité

**Avantages:** Capture la dynamique temporelle et l'hétérogénéité des pays.

**Note:** Cette implémentation utilise PanelOLS avec effets fixes, pas GMM. Les estimations peuvent souffrir du biais de Nickell (Nickell bias) avec T < 20.

---

### 4. Visualisations

Le système génère 9 visualisations professionnelles:

1. **Model Comparison Chart** - Comparaison des performances (R², MAE, RMSE)
2. **Feature Importance** - Variables les plus influentes
3. **Predictions vs Actual** - Scatter plots pour chaque modèle
4. **Statistical Analysis** - Distribution et corrélations
5. **Residual Plots** - Diagnostics des erreurs
6. **Time Series Predictions** - Évolution temporelle 2018-2023
7. **Cross-Validation Scores** - Robustesse des modèles
8. **Error Distribution** - Analyse des erreurs de prédiction
9. **Regional Analysis** - Performance par région géographique

---

## Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation des dépendances

```bash
# Cloner le projet
cd Datascience-and-Advanced-Programming-2025-2026-project-JK

# Installer les packages
pip install -r requirements.txt
```

**Packages principaux:**
- `pandas`, `numpy` - Manipulation de données
- `scikit-learn` - Modèles ML
- `xgboost` - Gradient boosting
- `linearmodels` - Panel data
- `matplotlib`, `seaborn` - Visualisations
- `streamlit` - Dashboard interactif

---

## Utilisation

### Option 1: Menu Interactif (Recommandé)

```bash
python3 main/main.py
```

#### Menu Principal

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POLITICAL STABILITY PREDICTION SYSTEM                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

[0]  Check Environment        -> Verify data files and dependencies
[1]  Run Data Preparation     -> Load, clean, and process raw data
[2]  Train Model              -> Train ML models or Panel Analysis
[3]  Display Test Results     -> Show model performance on 2018-2023
[4]  Generate Visualizations  -> Create charts and plots
[5]  Show Dashboard           -> Launch Streamlit dashboard
[Q]  Quit
```

#### Workflow Complet

**Étape 1: Vérifier l'environnement**
```bash
Choix: 0
```
Vérifie que tous les fichiers de données sont présents.

**Étape 2: Préparer les données**
```bash
Choix: 1
```
Charge, nettoie et traite les données brutes.
Crée `data/processed/train_data.csv` et `test_data.csv`.

**Étape 3: Entraîner les modèles**
```bash
Choix: 2
```

Options disponibles:
```
[1]  Train ALL (7 ML + Panel)     -> Benchmark complet (~5-10 min)
[2]  Dynamic Panel Analysis only  -> Modèle économétrique (~30s)
[3]  Random Forest                -> Meilleur ML (~2 min)
[4]  XGBoost                      -> Gradient boosting (~15s)
[5]  Gradient Boosting            -> Alternative (~1 min)
[6]  Elastic Net                  -> Régression linéaire (~5s)
[7]  SVR                          -> Support Vector (~10s)
[8]  KNN                          -> K-Nearest Neighbors (~2s)
[9]  MLP                          -> Neural Network (~30s)
```

**Recommandation:** Commencer par [1] pour comparer tous les modèles.

**Étape 4: Visualiser les résultats**
```bash
Choix: 3  # Afficher les performances
```

Tableau de classement professionnel avec barres visuelles:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MODEL PERFORMANCE RANKING                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Rank │      Model       │  R² Score    │    MAE     │    RMSE    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ***  1  │ Random Forest    │ 0.7611 ████████ │  0.3617    │  0.4670    ║
║ **   2  │ XGBoost          │ 0.7362 ███████░ │  0.3793    │  0.4909    ║
║ *    3  │ Gradient Boost   │ 0.7282 ███████░ │  0.3865    │  0.4982    ║
║      4  │ KNN              │ 0.6863 ██████░░ │  0.4098    │  0.5353    ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                       PERFORMANCE STATISTICS                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  Champion:         Random Forest              R² = 0.7611                   │
│  Average R²:       0.6850 ± 0.0512                                          │
│  Average MAE:      0.4159                                                   │
│  Models tested:    7/7                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Étape 5: Générer les visualisations**
```bash
Choix: 4
```

Options:
```
[1]   Model Comparison Chart      -> Compare R², MAE, RMSE
[2]   Feature Importance          -> Most important features
[3]   Predictions vs Actual       -> Scatter plots
[4]   Statistical Analysis        -> Distribution & correlation
[5]   Residual Plots              -> Model diagnostic plots
[6]   Learning Curves             -> Overfitting/underfitting (slow)
[7]   Time Series Predictions     -> Temporal evolution
[8]   Cross-Validation Scores     -> Model robustness
[9]   Error Distribution          -> Prediction errors analysis
[10]  Regional Analysis           -> Geographic performance
[11]  ALL VISUALIZATIONS          -> Generate 9 plots (fast)
```

**Recommandation:** [11] pour générer toutes les visualisations (~2-3 min).

**Étape 6: Dashboard interactif**
```bash
Choix: 5
```

Lance Streamlit dashboard sur `http://localhost:8501`

---

### Option 2: Dashboard Streamlit Direct

```bash
streamlit run main/dashboard.py
```

Le dashboard permet:
- Navigation interactive entre les résultats
- Comparaison visuelle des modèles
- Exploration des prédictions par pays
- Analyse temporelle 2018-2023
- Export des graphiques

---

### Option 3: Utilisation Programmatique

```python
from src.data_loader import load_data
from src.models import RandomForestPredictor
from pathlib import Path

# Charger les données
data_dict = load_data(
    data_path=Path('data/raw'),
    target='political_stability',
    train_end_year=2017
)

# Entraîner un modèle
model = RandomForestPredictor()
model.fit(data_dict['X_train'], data_dict['y_train'])

# Prédire et évaluer
metrics = model.evaluate(data_dict['X_test'], data_dict['y_test'])
print(f"R² Score: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

---

## Tests

Le projet inclut **67 tests unitaires** couvrant:
- Chargement et nettoyage des données
- Entraînement et prédiction de chaque modèle
- Métriques d'évaluation
- Filtrage par pays
- Panel data analysis

```bash
# Lancer tous les tests
pytest tests/test.py -v

# Lancer un test spécifique
pytest tests/test.py::TestRandomForestPredictor -v

# Avec rapport de couverture
pytest tests/test.py --cov=src --cov-report=html
```

---

## Résultats Typiques

### Performance des Modèles (Test 2018-2023)

| Modèle | R² | MAE | RMSE | Temps |
|--------|-----|-----|------|-------|
| **Random Forest** | **0.7611** | 0.3617 | 0.4670 | 103s |
| **XGBoost** | 0.7362 | 0.3793 | 0.4909 | 15s |
| **Gradient Boosting** | 0.7282 | 0.3865 | 0.4982 | 60s |
| **KNN** | 0.6863 | 0.4098 | 0.5353 | 2s |
| **MLP** | 0.6538 | 0.4246 | 0.5623 | 30s |
| **Elastic Net** | 0.6273 | 0.4663 | 0.5834 | 5s |
| **SVR** | 0.6224 | 0.4619 | 0.5872 | 8s |

### Features les Plus Importantes

1. **Rule of Law** (État de droit)
2. **Government Effectiveness** (Efficacité gouvernementale)
3. **GDP per Capita** (PIB par habitant)
4. **Unemployment** (Chômage)
5. **Political Stability (lag 1)** (Stabilité passée)

---

## Structure des Données

### Features (Prédicteurs)
- **Économiques:** PIB/habitant, croissance, inflation, chômage, commerce
- **Sociaux:** HDI
- **Gouvernance:** Rule of Law, Government Effectiveness

### Target (Variable à prédire)
- **Political Stability:** Score de -2.5 (très instable) à +2.5 (très stable)

### Format des Données
```
Index: (Country Name, Year)
Colonnes: [gdp_per_capita, unemployment, inflation, gdp_growth,
           effectiveness, rule_of_law, trade, hdi,
           political_stability]
```

---

## Fichiers de Sortie

Après exécution, les fichiers suivants sont créés:

### `data/processed/`
- `train_data.csv` - Données d'entraînement (2000-2017)
- `test_data.csv` - Données de test (2018-2023)
- `full_data.csv` - Dataset complet

### `results/`
- `benchmark_results.csv` - Performance de tous les modèles
- `panel_analysis.txt` - Résultats du modèle Dynamic Panel
- `ml_*_results.txt` - Résultats détaillés par modèle

### `results/figures/`
- `model_comparison_*.png`
- `feature_importance_*.png`
- `predictions_vs_actual_*.png`
- `residual_plots_*.png`
- `time_series_predictions_*.png`
- ... (9 visualisations au total)

---

## Troubleshooting

### Erreur: "No such file or directory: main.py"
```bash
# Solution: main.py est dans le sous-dossier main/
python3 main/main.py  # Correct
```

### Erreur: Module not found
```bash
# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
pip list | grep -E "pandas|scikit-learn|xgboost"
```

### Les visualisations sont lentes
- Learning Curves est très lent (désactivé par défaut dans "Generate All")
- Pour générer individuellement: Menu [4] puis [6]
- Temps estimé: ~5-10 minutes pour Learning Curves

### Données manquantes
```bash
# Vérifier les fichiers
ls data/raw/

# Devrait contenir:
# - API_*.csv (World Bank)
# - HDI.xlsx (UNDP)
```

---

## Technologies Utilisées

- **Python 3.12**
- **Machine Learning:** scikit-learn, XGBoost
- **Data Processing:** pandas, numpy
- **Panel Data:** linearmodels (PanelOLS)
- **Visualization:** matplotlib, seaborn
- **Dashboard:** Streamlit
- **Testing:** pytest

---

## Licence

Usage académique uniquement.
