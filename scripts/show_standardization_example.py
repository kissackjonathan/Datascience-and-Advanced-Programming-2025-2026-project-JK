"""
Show concrete example of standardization on actual data
Display before/after comparison for better understanding
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 100)
print("EXEMPLE CONCRET DE STANDARDISATION")
print("=" * 100)
print()

# Load data
df = pd.read_csv(data_dir / 'final_clean_data.csv')

# Select features
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

# Take first 10 rows as example
X_sample = X.head(10).copy()

print("DONNEES ORIGINALES (10 premieres lignes):")
print("=" * 100)
print(X_sample.to_string())
print()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
X_scaled_df = pd.DataFrame(X_scaled, columns=predictor_cols, index=X_sample.index)

print()
print("DONNEES STANDARDISEES (memes 10 lignes):")
print("=" * 100)
print(X_scaled_df.to_string())
print()

# Show statistics
print()
print("=" * 100)
print("STATISTIQUES COMPARATIVES")
print("=" * 100)
print()

print("AVANT STANDARDISATION:")
print("-" * 100)
print(X_sample.describe().loc[['mean', 'std', 'min', 'max']].to_string())
print()

print()
print("APRES STANDARDISATION:")
print("-" * 100)
print(X_scaled_df.describe().loc[['mean', 'std', 'min', 'max']].to_string())
print()

# Show transformation for one specific row
print()
print("=" * 100)
print("EXEMPLE DETAILLE: TRANSFORMATION DE LA PREMIERE LIGNE")
print("=" * 100)
print()

print(f"{'Variable':<25} {'Valeur Originale':>20} {'Moyenne':>15} {'Ecart-type':>15} {'Valeur Standardisee':>20}")
print("-" * 100)

for col in predictor_cols:
    original_value = X_sample.iloc[0][col]
    mean = X_sample[col].mean()
    std = X_sample[col].std()
    standardized_value = X_scaled_df.iloc[0][col]

    print(f"{col:<25} {original_value:>20.2f} {mean:>15.2f} {std:>15.2f} {standardized_value:>20.2f}")

print()
print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()
print("Formule appliquee: z = (x - mean) / std")
print()
print("Exemple avec gdp_per_capita (premiere ligne):")
mean_gdp = X_sample['gdp_per_capita'].mean()
std_gdp = X_sample['gdp_per_capita'].std()
val_gdp = X_sample.iloc[0]['gdp_per_capita']
z_gdp = (val_gdp - mean_gdp) / std_gdp
print(f"  Valeur originale: {val_gdp:.2f}")
print(f"  Moyenne: {mean_gdp:.2f}")
print(f"  Ecart-type: {std_gdp:.2f}")
print(f"  z = ({val_gdp:.2f} - {mean_gdp:.2f}) / {std_gdp:.2f} = {z_gdp:.2f}")
print()
print("Interpretation:")
print(f"  z = {z_gdp:.2f} signifie que cette valeur est a {abs(z_gdp):.2f} ecarts-types")
if z_gdp > 0:
    print(f"  AU-DESSUS de la moyenne")
else:
    print(f"  EN-DESSOUS de la moyenne")
print()

# Show scale comparison for all variables
print()
print("=" * 100)
print("COMPARAISON DES ECHELLES")
print("=" * 100)
print()
print(f"{'Variable':<25} {'Min Original':>15} {'Max Original':>15} {'Etendue':>15} {'Min Std':>15} {'Max Std':>15}")
print("-" * 100)

for col in predictor_cols:
    min_orig = X_sample[col].min()
    max_orig = X_sample[col].max()
    range_orig = max_orig - min_orig
    min_std = X_scaled_df[col].min()
    max_std = X_scaled_df[col].max()

    print(f"{col:<25} {min_orig:>15.2f} {max_orig:>15.2f} {range_orig:>15.2f} {min_std:>15.2f} {max_std:>15.2f}")

print()
print()
print("=" * 100)
print("AVANTAGES DE LA STANDARDISATION")
print("=" * 100)
print()
print("1. MEME ECHELLE pour toutes les variables")
print("   - Avant: gdp_per_capita varie de 109 a 134,000")
print("   - Avant: hdi varie de 0.2 a 1.0")
print("   - Apres: toutes les variables varient entre -3 et +3 (environ)")
print()
print("2. MOYENNE = 0 et ECART-TYPE = 1")
print("   - Facilite la comparaison entre variables")
print("   - Necessite pour certains algorithmes (SVR, Ridge, Lasso)")
print()
print("3. PRESERVE LES RELATIONS")
print("   - Les correlations entre variables restent identiques")
print("   - Seule l'echelle change, pas la structure des donnees")
print()
