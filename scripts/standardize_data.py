"""
Standardize Final Clean Data
Apply z-score standardization to features and save standardized dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

data_dir = Path(__file__).parent.parent / "data" / "processed"

print("=" * 80)
print("STANDARDIZATION DES DONNEES")
print("=" * 80)
print()

# Load clean data
df = pd.read_csv(data_dir / 'final_clean_data.csv')
print(f"Donnees chargees: {len(df)} lignes, {df['Country Name'].nunique()} pays")
print(f"Annees: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print()

# Define features and target
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

# Metadata columns to preserve
metadata_cols = ['Country Name', 'Country Code', 'Year']

print("Variables a standardiser:")
for i, col in enumerate(predictor_cols, 1):
    print(f"  {i}. {col}")
print()

print(f"Variable cible (NON standardisee): {target_col}")
print()

# Separate features, target, and metadata
X = df[predictor_cols].copy()
y = df[target_col].copy()
metadata = df[metadata_cols].copy()

# Statistics BEFORE standardization
print("=" * 80)
print("STATISTIQUES AVANT STANDARDISATION")
print("=" * 80)
print()
print(X.describe().loc[['mean', 'std', 'min', 'max']].to_string())
print()

# Standardize features using z-score
print("=" * 80)
print("APPLICATION DU Z-SCORE")
print("=" * 80)
print()
print("Methode: StandardScaler (sklearn)")
print("Formule: z = (x - mean) / std")
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=predictor_cols, index=X.index)

# Statistics AFTER standardization
print("=" * 80)
print("STATISTIQUES APRES STANDARDISATION")
print("=" * 80)
print()
print(X_scaled_df.describe().loc[['mean', 'std', 'min', 'max']].to_string())
print()

# Verify standardization
print("=" * 80)
print("VERIFICATION DE LA STANDARDISATION")
print("=" * 80)
print()
print("Toutes les variables doivent avoir:")
print("  - Moyenne proche de 0 (tolerance: < 1e-10)")
print("  - Ecart-type proche de 1 (tolerance: < 0.01)")
print()

all_good = True
for col in predictor_cols:
    mean = X_scaled_df[col].mean()
    std = X_scaled_df[col].std()
    mean_ok = abs(mean) < 1e-10
    std_ok = abs(std - 1.0) < 0.01

    status_mean = "OK" if mean_ok else "ERREUR"
    status_std = "OK" if std_ok else "ERREUR"

    print(f"{col:30s}: mean={mean:>12.2e} [{status_mean:>6}]  std={std:>6.4f} [{status_std:>6}]")

    if not (mean_ok and std_ok):
        all_good = False

print()
if all_good:
    print("Standardisation validee: toutes les variables sont correctement standardisees")
else:
    print("ATTENTION: Certaines variables ne sont pas correctement standardisees")
print()

# Reconstruct full dataset with standardized features
print("=" * 80)
print("RECONSTRUCTION DU DATASET COMPLET")
print("=" * 80)
print()

# Combine metadata, standardized features, and target
df_standardized = pd.concat([
    metadata.reset_index(drop=True),
    X_scaled_df.reset_index(drop=True),
    y.reset_index(drop=True)
], axis=1)

print(f"Dataset standardise: {df_standardized.shape[0]} lignes x {df_standardized.shape[1]} colonnes")
print()
print("Colonnes:")
print(f"  - Metadata: {metadata_cols}")
print(f"  - Features standardisees: {predictor_cols}")
print(f"  - Target (non standardisee): {target_col}")
print()

# Save standardized data
output_file = data_dir / 'final_clean_data_standardized.csv'
df_standardized.to_csv(output_file, index=False)

print("=" * 80)
print("SAUVEGARDE DES FICHIERS")
print("=" * 80)
print()
print(f"Donnees standardisees sauvegardees: {output_file.name}")
print(f"  Lignes: {len(df_standardized)}")
print(f"  Colonnes: {len(df_standardized.columns)}")
print(f"  Taille: {output_file.stat().st_size / 1024:.1f} KB")
print()

# Save scaler for future use
scaler_file = data_dir / 'scaler.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Scaler sauvegarde: {scaler_file.name}")
print(f"  Usage: scaler = pickle.load(open('{scaler_file.name}', 'rb'))")
print(f"  Permet de standardiser de nouvelles donnees avec les memes parametres")
print()

# Save scaler parameters for reference
scaler_params = pd.DataFrame({
    'feature': predictor_cols,
    'mean': scaler.mean_,
    'std': scaler.scale_
})

scaler_params_file = data_dir / 'scaler_parameters.csv'
scaler_params.to_csv(scaler_params_file, index=False)

print(f"Parametres du scaler: {scaler_params_file.name}")
print()
print(scaler_params.to_string(index=False))
print()

# Summary
print("=" * 80)
print("RESUME")
print("=" * 80)
print()
print("Fichiers crees:")
print(f"  1. {output_file.name}")
print(f"     -> Dataset avec features standardisees (z-score)")
print(f"     -> {len(df_standardized)} observations")
print()
print(f"  2. {scaler_file.name}")
print(f"     -> Objet StandardScaler (pickle)")
print(f"     -> Pour standardiser de nouvelles donnees")
print()
print(f"  3. {scaler_params_file.name}")
print(f"     -> Parametres du scaler (mean, std)")
print(f"     -> Pour reference")
print()
print("Utilisation recommandee:")
print("  - Pour ML avec SVR, Ridge, Lasso: utiliser final_clean_data_standardized.csv")
print("  - Pour Random Forest, Gradient Boosting: utiliser final_clean_data.csv")
print("  - Pour comparaison: tester les deux versions")
print()
print("=" * 80)
print("DONE")
print("=" * 80)
