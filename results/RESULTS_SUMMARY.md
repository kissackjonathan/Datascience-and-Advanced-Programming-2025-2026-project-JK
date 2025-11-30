# Dynamic Panel Analysis Results
## Political Stability Prediction (1996-2023)

**Analysis Date:** November 30, 2025
**Model:** Dynamic Panel with Two-Way Fixed Effects
**Target Variable:** Political Stability Index

---

## 📊 Data Overview

- **Total Observations:** 4,150
- **Countries:** 166
- **Time Period:** 1996-2023 (25 years)
- **Training Period:** 1996-2017 (22 years, 3,154 obs)
- **Test Period:** 2018-2023 (6 years, 996 obs)

---

## 🎯 Model Performance

### Training Set Performance (1996-2017)

| Model | R² (within) | Observations |
|-------|-------------|--------------|
| Fixed Effects | 0.1586 | 3,154 |
| Random Effects | 0.2277 | 3,154 |
| **Dynamic Panel** | **0.6049** | 2,988 |

### Test Set Performance (2018-2023)

| Metric | Value |
|--------|-------|
| **Test R²** | **0.6736** |
| **RMSE** | **0.5392** |
| **Test Observations** | 830 |

---

## 📈 Model Coefficients (Dynamic Panel)

### Significant Predictors (p < 0.05)

| Variable | Coefficient | Std. Error | T-stat | P-value | Significance |
|----------|-------------|------------|--------|---------|--------------|
| **political_stability_lag1** | **0.6935** | 0.0276 | 25.15 | 0.000 | *** |
| **gdp_growth** | **0.0060** | 0.0015 | 4.04 | 0.000 | *** |
| **rule_of_law** | **0.1550** | 0.0488 | 3.17 | 0.002 | *** |
| **government_effectiveness** | **0.0861** | 0.0326 | 2.64 | 0.008 | *** |

### Non-Significant Predictors (p ≥ 0.05)

| Variable | Coefficient | Std. Error | T-stat | P-value |
|----------|-------------|------------|--------|---------|
| gdp_per_capita | -5.13e-07 | 9.11e-07 | -0.56 | 0.574 |
| unemployment_ilo | -0.0036 | 0.0023 | -1.52 | 0.128 |
| inflation_cpi | -0.0001 | 0.0005 | -0.25 | 0.802 |
| trade_gdp_pct | -0.0007 | 0.0004 | -1.87 | 0.062 |
| hdi | 0.7219 | 0.4902 | 1.47 | 0.141 |

**Significance levels:** `***` p<0.01, `**` p<0.05, `*` p<0.1

---

## 🔍 Diagnostic Tests

### Hausman Test (FE vs RE)
- **Test Statistic:** χ² = 11.02
- **P-value:** 0.200
- **Decision:** Use Random Effects (but using FE for conservative approach)

### Breusch-Pagan Test (Heteroskedasticity)
- **LM Statistic:** 207.72
- **P-value:** < 0.001
- **Decision:** **Heteroskedasticity detected** - use robust standard errors ✓

### Wooldridge Test (Serial Correlation)
- **Test Statistic:** -7.93
- **P-value:** < 0.001
- **Decision:** **Serial correlation detected** - use clustered standard errors ✓

---

## 📊 Persistence Analysis

### Lagged Dependent Variable

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Persistence Coefficient (ρ)** | **0.6935** | MODERATE persistence |
| **Half-life of Shocks** | **1.89 years** | Shocks decay by 50% in ~2 years |
| **P-value** | < 0.001 | Highly significant |

### Interpretation

The persistence coefficient of **ρ = 0.6935** indicates:
- Political stability exhibits **moderate persistence** over time
- 69.35% of current stability is explained by previous year's stability
- Shocks to political stability decay with a half-life of approximately **1.89 years**
- Full decay (to <5% of original shock) takes approximately **5-6 years**

---

## 💡 Key Findings

### 1. Lagged Stability (ρ = 0.6935, p < 0.001)
- **Most important predictor** by far (t-stat = 25.15)
- Countries with stable politics tend to remain stable
- Countries experiencing instability face prolonged periods of uncertainty

### 2. GDP Growth (β = 0.0060, p < 0.001)
- Economic growth **positively** associated with political stability
- 1% increase in GDP growth → +0.006 units increase in stability
- Confirms economic prosperity-stability nexus

### 3. Rule of Law (β = 0.1550, p = 0.002)
- Strong legal institutions **significantly** improve stability
- 1-unit increase in rule of law → +0.155 units stability increase
- Second most important time-varying predictor

### 4. Government Effectiveness (β = 0.0861, p = 0.008)
- Effective governance enhances political stability
- Complements rule of law in institutional quality

### 5. Non-Significant Factors
- **GDP per capita:** No significant effect (once growth is controlled)
- **Unemployment:** Negative but not significant
- **Inflation:** No significant effect
- **Trade openness:** Marginally negative, not significant
- **HDI:** Positive but not significant

---

## 🎯 Model Validation

### Out-of-Sample Performance
- **Test R² = 0.6736:** Model explains 67% of variance in unseen data
- **Test RMSE = 0.5392:** Average prediction error of ~0.54 units
- **Strong generalization:** Performance consistent between train and test

### Robustness Checks
- ✅ Clustered standard errors (by country)
- ✅ Two-way fixed effects (entity + time)
- ✅ Appropriate handling of heteroskedasticity
- ✅ Serial correlation accounted for

---

## 📝 Methodology

### Model Specification

**Dynamic Panel Model with Two-Way Fixed Effects:**

```
y_it = ρ * y_i,t-1 + β₁ * GDP_growth_it + β₂ * Rule_of_law_it +
       β₃ * Gov_effectiveness_it + ... + α_i + λ_t + ε_it
```

Where:
- `y_it`: Political stability for country i at time t
- `y_i,t-1`: Lagged political stability (persistence)
- `α_i`: Country fixed effects (time-invariant heterogeneity)
- `λ_t`: Time fixed effects (common shocks)
- `ε_it`: Idiosyncratic error term

### Estimation Method
- **Estimator:** Panel OLS with Two-Way Fixed Effects
- **Standard Errors:** Clustered at country level
- **Observations:** 2,988 (after lagging and panel structure)

---

## 📌 Recommendations

### For Policy Makers
1. **Institutional Quality:** Invest in rule of law and government effectiveness
2. **Economic Growth:** Promote sustainable economic development
3. **Long-term Perspective:** Political stability takes years to build or erode

### For Researchers
1. **Persistence Matters:** Always include lagged dependent variable
2. **Fixed Effects:** Control for unobserved country heterogeneity
3. **Robust Inference:** Use clustered standard errors

### For Forecasting
1. **Past is Prologue:** Historical stability is the best predictor
2. **Economic Indicators:** Monitor GDP growth trends
3. **Institutional Metrics:** Track rule of law indices

---

## 📚 Technical Notes

### Software Stack
- **Python 3.9+**
- **linearmodels:** Panel data econometrics
- **pandas:** Data manipulation
- **statsmodels:** Diagnostic tests

### Code Repository
- **Main Script:** `main.py`
- **Model Class:** `src/models/panel_models.py`
- **Logging:** `src/utils/logging_config.py`

### Reproducibility
```bash
python3 main.py --mode dynamic_panel --train_end_year 2017
```

---

## 📞 Contact

For questions about this analysis:
- **Author:** Jonathan Kissack
- **Institution:** EPFL
- **Course:** Data Science and Advanced Programming 2025-2026
- **Model:** Dynamic Panel Analysis with Claude Code

---

**Report Generated:** November 30, 2025
**Analysis Tool:** Claude Code by Anthropic
**License:** Educational Use
