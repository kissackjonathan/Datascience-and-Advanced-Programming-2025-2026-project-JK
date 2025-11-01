# Datascience-and-Advanced-Programming-2025-2026-project-JK
Project Title:
Predicting Political Stability Using Economic and Social Indicators
Category:
Data Analysis & Machine Learning (Geopolitics / Economics)

Problem Statement or Motivation
Political stability is fundamental for a country — especially for its leadership, investment climate and the welfare of its citizens. Recently, we have witnessed major protests and regime change in countries such as Madagascar in late 2025, where youth-led demonstrations and a military takeover followed widespread dissatisfaction over basic services and governance. Similarly, in Nepal in early September 2025, mass protests brought down the government amid frustration with corruption, social media bans and elite capture.These events highlight how quickly stability can be eroded. This trend could see Tanzania’s leaders follow the same road very soon. The core problem of this project is: Which economic, social and institutional variables influence political instability — and can they be used to predict future instability? This question motivates me deeply: being passionate about geopolitics, I am fascinated by how major changes in a country’s political situation may be explained, or even anticipated, through social/economic data trends.

Planned Approach and Technologies
I will draw on open international datasets:
* Fund for Peace’s Fragile States Index (FSI) will serve as the target variable — a yearly stability score per country.
* World Bank Open Data will provide features such as GDP per capita, unemployment, trade openness, inflation and public debt.
* United Nations Development Programme (UNDP) Human Development Index will supply education, health and income components (social variables). The workflow will begin with data acquisition and cleaning, followed by exploratory visualisation to detect patterns and correlations. Next, I will train and compare various machine learning models (I do not know which one yet) to estimate each country’s stability score.( As a stretch goal I may also include Difference-in-Differences (DiD) analysis and Monte Carlo simulations to explore causal impacts of major shocks and scenario-based future instability risk.)

Expected Challenges and How I’ll Address Them
* Country selection and coverage: I may not use all countries if data quality varies. I will select a subset of countries with reliable data across years to ensure consistency.
* Data quality and missing values: Different sources may have gaps ,misalignments or the trend is not explaining what it should be.
* Model choice and interpretability: Choosing the right machine learning model is crucial, ill have to do some more research on the different machine learning models.

Success Criteria
* Machine learning accuracy >80% (if classification into stable vs unstable).
* Clear visualisations and feature‐importance plots demonstrating which variables drive instability.
* A reproducible codebase with documentation and notebooks that support the project’s findings.

Stretch Goals (if time permits)
* Develop an interactive dashboard (Streamlit) where users explore predicted stability by country and scenario.
* Extend the model to predict future probability of instability (e.g., for 2026-2030) using Monte Carlo scenario simulations.
