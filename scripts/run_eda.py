"""
Exploratory Data Analysis (EDA) Script
Political Stability Prediction Project

Generates comprehensive EDA with visualizations saved to results/eda/
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

import warnings
warnings.filterwarnings('ignore')

# Paths
data_dir = Path(__file__).parent.parent / "data" / "processed"
results_dir = Path(__file__).parent.parent / "results" / "eda"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("Political Stability Prediction Project")
print("=" * 100)
print()

# Load dataset
df = pd.read_csv(data_dir / 'final_clean_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Number of countries: {df['Country Name'].nunique()}")
print(f"Time period: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print(f"Total observations: {len(df):,}")
print()

# Define features and target
target_col = 'political_stability'
feature_cols = [
    'gdp_per_capita',
    'gdp_growth',
    'unemployment_ilo',
    'inflation_cpi',
    'trade_gdp_pct',
    'rule_of_law',
    'government_effectiveness',
    'hdi'
]

all_numeric_cols = feature_cols + [target_col]

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("=" * 100)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 100)
print()

stats_summary = pd.DataFrame({
    'mean': df[all_numeric_cols].mean(),
    'median': df[all_numeric_cols].median(),
    'std': df[all_numeric_cols].std(),
    'skewness': df[all_numeric_cols].skew(),
    'kurtosis': df[all_numeric_cols].kurtosis(),
    'min': df[all_numeric_cols].min(),
    'max': df[all_numeric_cols].max()
})

print(stats_summary.to_string())
print()

# Save statistics
stats_summary.to_csv(results_dir / 'descriptive_statistics.csv')
print(f"✓ Saved: descriptive_statistics.csv")
print()

# ============================================================================
# 2. DISTRIBUTION ANALYSIS - HISTOGRAMS
# ============================================================================
print("=" * 100)
print("2. DISTRIBUTION ANALYSIS - Histograms")
print("=" * 100)
print()

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, col in enumerate(all_numeric_cols):
    ax = axes[idx]

    # Histogram
    ax.hist(df[col].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')

    # Add mean and median lines
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'distributions_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: distributions_histograms.png")
print()

# ============================================================================
# 3. BOX PLOTS - OUTLIER DETECTION
# ============================================================================
print("=" * 100)
print("3. BOX PLOTS - Outlier Detection")
print("=" * 100)
print()

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, col in enumerate(all_numeric_cols):
    ax = axes[idx]

    # Box plot
    bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='blue'),
                     medianprops=dict(color='red', linewidth=2))

    ax.set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
    ax.set_ylabel(col, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add outlier count
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    ax.text(0.5, 0.95, f'Outliers: {len(outliers)}',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(results_dir / 'boxplots_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: boxplots_outliers.png")
print()

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("=" * 100)
print("4. CORRELATION ANALYSIS")
print("=" * 100)
print()

# Correlation matrix
corr_matrix = df[all_numeric_cols].corr()

print("Correlation with Political Stability (Target):")
print("-" * 100)
correlations_with_target = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
print(correlations_with_target.to_string())
print()

# Save correlations
correlations_with_target.to_csv(results_dir / 'correlations_with_target.csv')

# Correlation heatmap
plt.figure(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})

plt.title('Correlation Matrix - Political Stability Features',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(results_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: correlation_heatmap.png")
print()

# Bar plot of correlations with target
plt.figure(figsize=(12, 6))

colors = ['green' if x > 0 else 'red' for x in correlations_with_target.values]
plt.barh(correlations_with_target.index, correlations_with_target.values, color=colors, alpha=0.7)
plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Correlation of Features with Political Stability', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for idx, value in enumerate(correlations_with_target.values):
    plt.text(value, idx, f' {value:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / 'correlation_barplot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: correlation_barplot.png")
print()

# ============================================================================
# 5. TIME SERIES ANALYSIS
# ============================================================================
print("=" * 100)
print("5. TIME SERIES ANALYSIS")
print("=" * 100)
print()

# Average political stability over time
stability_by_year = df.groupby('Year')[target_col].agg(['mean', 'std', 'count'])

plt.figure(figsize=(14, 6))

plt.plot(stability_by_year.index, stability_by_year['mean'],
         marker='o', linewidth=2, markersize=6, color='steelblue', label='Mean')

# Add confidence interval
plt.fill_between(stability_by_year.index,
                  stability_by_year['mean'] - stability_by_year['std'],
                  stability_by_year['mean'] + stability_by_year['std'],
                  alpha=0.2, color='steelblue', label='±1 Std Dev')

plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Political Stability (Mean)', fontsize=12, fontweight='bold')
plt.title('Global Political Stability Trend (1996-2023)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(results_dir / 'time_series_global_stability.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: time_series_global_stability.png")
print()

# Evolution of all features over time
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, col in enumerate(all_numeric_cols):
    ax = axes[idx]

    yearly_stats = df.groupby('Year')[col].agg(['mean', 'std'])

    ax.plot(yearly_stats.index, yearly_stats['mean'],
            marker='o', linewidth=2, markersize=4, color='darkgreen')

    ax.fill_between(yearly_stats.index,
                     yearly_stats['mean'] - yearly_stats['std'],
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.2, color='darkgreen')

    ax.set_title(f'Evolution of {col}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'time_series_all_features.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: time_series_all_features.png")
print()

# ============================================================================
# 6. GEOGRAPHIC ANALYSIS
# ============================================================================
print("=" * 100)
print("6. GEOGRAPHIC ANALYSIS")
print("=" * 100)
print()

# Country-level averages
country_stability = df.groupby('Country Name')[target_col].mean().sort_values(ascending=False)

print("Top 10 Most Stable Countries:")
print("-" * 100)
print(country_stability.head(10).to_string())
print()

print("Top 10 Least Stable Countries:")
print("-" * 100)
print(country_stability.tail(10).to_string())
print()

# Save country rankings
country_stability.to_csv(results_dir / 'country_stability_rankings.csv')

# Visualize top/bottom countries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Top 15 most stable
top_15 = country_stability.head(15)
ax1.barh(range(len(top_15)), top_15.values, color='green', alpha=0.7)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15.index, fontsize=9)
ax1.set_xlabel('Average Political Stability', fontsize=11, fontweight='bold')
ax1.set_title('Top 15 Most Stable Countries', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Bottom 15 least stable
bottom_15 = country_stability.tail(15)
ax2.barh(range(len(bottom_15)), bottom_15.values, color='red', alpha=0.7)
ax2.set_yticks(range(len(bottom_15)))
ax2.set_yticklabels(bottom_15.index, fontsize=9)
ax2.set_xlabel('Average Political Stability', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Least Stable Countries', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(results_dir / 'geographic_country_rankings.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: geographic_country_rankings.png")
print()

# Time series for selected countries
selected_countries = [
    'United States',
    'France',
    'China',
    'Brazil',
    'India',
    'South Africa',
    'Japan',
    'Nigeria'
]

plt.figure(figsize=(14, 7))

for country in selected_countries:
    country_data = df[df['Country Name'] == country].sort_values('Year')
    if len(country_data) > 0:
        plt.plot(country_data['Year'], country_data[target_col],
                 marker='o', linewidth=2, markersize=4, label=country)

plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Political Stability', fontsize=12, fontweight='bold')
plt.title('Political Stability Trends - Selected Countries', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / 'geographic_selected_countries.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: geographic_selected_countries.png")
print()

# ============================================================================
# 7. BIVARIATE ANALYSIS
# ============================================================================
print("=" * 100)
print("7. BIVARIATE ANALYSIS - Features vs Target")
print("=" * 100)
print()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, feature in enumerate(feature_cols):
    ax = axes[idx]

    # Scatter plot
    ax.scatter(df[feature], df[target_col], alpha=0.3, s=10, color='steelblue')

    # Add regression line
    z = np.polyfit(df[feature].dropna(), df[target_col][df[feature].notna()], 1)
    p = np.poly1d(z)
    ax.plot(df[feature].sort_values(), p(df[feature].sort_values()),
            "r--", linewidth=2, label='Trend line')

    # Correlation coefficient
    corr_val = df[[feature, target_col]].corr().iloc[0, 1]

    ax.set_xlabel(feature, fontsize=10, fontweight='bold')
    ax.set_ylabel('Political Stability', fontsize=10, fontweight='bold')
    ax.set_title(f'{feature} vs Political Stability\n(r = {corr_val:.3f})',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig(results_dir / 'bivariate_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: bivariate_scatter_plots.png")
print()

# ============================================================================
# 8. OUTLIER ANALYSIS
# ============================================================================
print("=" * 100)
print("8. OUTLIER ANALYSIS (IQR Method)")
print("=" * 100)
print()

outlier_summary = []

for col in all_numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df)) * 100

    outlier_summary.append({
        'Variable': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'N_Outliers': n_outliers,
        'Pct_Outliers': pct_outliers
    })

outlier_df = pd.DataFrame(outlier_summary)

print(outlier_df.to_string(index=False))
print()

# Save outlier analysis
outlier_df.to_csv(results_dir / 'outlier_analysis.csv', index=False)
print(f"✓ Saved: outlier_analysis.csv")
print()

# Visualize outlier counts
plt.figure(figsize=(12, 6))

plt.bar(outlier_df['Variable'], outlier_df['N_Outliers'],
        color='coral', alpha=0.7, edgecolor='black')

plt.xlabel('Variable', fontsize=12, fontweight='bold')
plt.ylabel('Number of Outliers', fontsize=12, fontweight='bold')
plt.title('Outlier Count by Variable (IQR Method)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for idx, row in outlier_df.iterrows():
    plt.text(idx, row['N_Outliers'], f"{row['Pct_Outliers']:.1f}%",
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / 'outlier_counts.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: outlier_counts.png")
print()

# ============================================================================
# 9. KEY INSIGHTS SUMMARY
# ============================================================================
print("=" * 100)
print("KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS")
print("=" * 100)
print()

insights = []

insights.append("=" * 100)
insights.append("KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS")
insights.append("=" * 100)
insights.append("")

insights.append("1. DATASET OVERVIEW")
insights.append("-" * 100)
insights.append(f"   • Total observations: {len(df):,}")
insights.append(f"   • Number of countries: {df['Country Name'].nunique()}")
insights.append(f"   • Time period: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
insights.append(f"   • Features: {len(feature_cols)} predictors + 1 target")
insights.append(f"   • Missing values: {df[all_numeric_cols].isnull().sum().sum()}")
insights.append("")

insights.append("2. TARGET VARIABLE (Political Stability)")
insights.append("-" * 100)
insights.append(f"   • Mean: {df[target_col].mean():.3f}")
insights.append(f"   • Median: {df[target_col].median():.3f}")
insights.append(f"   • Std Dev: {df[target_col].std():.3f}")
insights.append(f"   • Range: [{df[target_col].min():.3f}, {df[target_col].max():.3f}]")
insights.append(f"   • Skewness: {df[target_col].skew():.3f}")
insights.append("")

insights.append("3. STRONGEST CORRELATIONS WITH POLITICAL STABILITY")
insights.append("-" * 100)
top_3_corr = correlations_with_target.head(3)
for idx, (feature, corr) in enumerate(top_3_corr.items(), 1):
    insights.append(f"   {idx}. {feature:30s}: r = {corr:+.3f}")
insights.append("")

insights.append("4. WEAKEST CORRELATIONS WITH POLITICAL STABILITY")
insights.append("-" * 100)
bottom_3_corr = correlations_with_target.tail(3)
for idx, (feature, corr) in enumerate(bottom_3_corr.items(), 1):
    insights.append(f"   {idx}. {feature:30s}: r = {corr:+.3f}")
insights.append("")

insights.append("5. OUTLIER PREVALENCE")
insights.append("-" * 100)
outlier_df_sorted = outlier_df.sort_values('N_Outliers', ascending=False)
for idx, row in outlier_df_sorted.head(3).iterrows():
    insights.append(f"   • {row['Variable']:30s}: {row['N_Outliers']:4.0f} outliers ({row['Pct_Outliers']:.1f}%)")
insights.append("")

insights.append("6. TEMPORAL TRENDS")
insights.append("-" * 100)
first_year_avg = df[df['Year'] == df['Year'].min()][target_col].mean()
last_year_avg = df[df['Year'] == df['Year'].max()][target_col].mean()
change = last_year_avg - first_year_avg
insights.append(f"   • Political Stability in {df['Year'].min():.0f}: {first_year_avg:.3f}")
insights.append(f"   • Political Stability in {df['Year'].max():.0f}: {last_year_avg:.3f}")
insights.append(f"   • Change: {change:+.3f} ({'increase' if change > 0 else 'decrease'})")
insights.append("")

insights.append("7. GEOGRAPHIC INSIGHTS")
insights.append("-" * 100)
insights.append(f"   • Most stable country: {country_stability.idxmax()} ({country_stability.max():.3f})")
insights.append(f"   • Least stable country: {country_stability.idxmin()} ({country_stability.min():.3f})")
insights.append(f"   • Stability range: {country_stability.max() - country_stability.min():.3f}")
insights.append("")

insights.append("=" * 100)
insights.append("EDA COMPLETE")
insights.append("=" * 100)

# Print and save insights
for line in insights:
    print(line)

with open(results_dir / 'eda_insights_summary.txt', 'w') as f:
    f.write('\n'.join(insights))

print()
print(f"✓ Saved: eda_insights_summary.txt")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 100)
print("EDA EXECUTION COMPLETE")
print("=" * 100)
print()
print(f"All results saved to: {results_dir}")
print()
print("Generated files:")
print("  1. descriptive_statistics.csv")
print("  2. correlations_with_target.csv")
print("  3. country_stability_rankings.csv")
print("  4. outlier_analysis.csv")
print("  5. eda_insights_summary.txt")
print()
print("Generated visualizations:")
print("  1. distributions_histograms.png")
print("  2. boxplots_outliers.png")
print("  3. correlation_heatmap.png")
print("  4. correlation_barplot.png")
print("  5. time_series_global_stability.png")
print("  6. time_series_all_features.png")
print("  7. geographic_country_rankings.png")
print("  8. geographic_selected_countries.png")
print("  9. bivariate_scatter_plots.png")
print(" 10. outlier_counts.png")
print()
print("=" * 100)
