# STAT 486 — Recession Prediction Project

**Team Members:** Michael Stutzman and Ryan Morrow

## Project Overview

Our project investigates whether machine learning can predict economic recessions in the United States using historical economic data.

We plan to collect economic indicators from the Federal Reserve Economic Data (FRED) database and combine them into a single dataset. We will then use different machine learning models to see whether these indicators can help predict future recessions.

**Research Question:** Can we use machine learning to predict recessions in the United States before they happen?

## Project Checkpoints

### October 27, 2026 — Project Proposal

We chose Route C, which involves building a dataset and using it in a reproducible machine learning pipeline.

We decided to use economic indicators from FRED, including unemployment, inflation, industrial production, and interest rates. We also plan to use historical recession data from NBER.

Our initial plan is to compare logistic regression, LASSO, and random forests. We will evaluate our models using a time-based train/test split and metrics such as precision, recall, and F1 score.

Some anticipated challenges include missing data, different reporting frequencies, and the relatively small number of historical recessions.

### November 5, 2026 — Data and Reference Result

Our current focus is collecting economic data from FRED and combining the indicators into a monthly dataset.

We are starting with unemployment, industrial production, inflation, Treasury yield spreads, and federal funds rates. We also plan to use the USREC series to identify recession periods.

Our first reference model will be logistic regression, predicting whether the economy will be in a recession three months into the future.

We plan to use earlier years for training and later years for testing to avoid data leakage associated with random splitting.

**Next Steps:**
- Finish cleaning and combining the data.
- Run our initial logistic regression.
- Compare the results with a simple baseline.
- Examine potential data quality issues.

### November 12, 2026 — Feature Engineering and Model Development

*Planned checkpoint*

We plan to explore additional economic indicators and create new features that may improve recession predictions.

This could include changes in unemployment, inflation rates, and lagged economic indicators.

We will also begin comparing logistic regression with LASSO to see whether feature selection improves model performance.

**Goals:**
- Explore relationships between economic indicators.
- Create additional predictor variables.
- Run LASSO logistic regression.
- Compare results with our initial model.

### November 19, 2026 — Model Comparison

*Planned checkpoint*

We plan to introduce a random forest model and compare it with our logistic regression and LASSO models.

We will examine precision, recall, F1 score, and ROC AUC to understand how well each model identifies recessions.

We also want to investigate false negatives, since missing an actual recession is particularly important for our research question.

**Goals:**
- Train a random forest model.
- Compare performance across models.
- Examine feature importance.
- Investigate incorrect predictions.

### November 26, 2026 — Model Improvements and Limitations

*Planned checkpoint*

We plan to use our previous results to improve the models and investigate their limitations.

We want to determine whether our models perform consistently across different economic periods and whether some indicators are more useful than others.

We will also consider how reporting delays, historical data revisions, and the limited number of recessions might affect our conclusions.

**Goals:**
- Improve model performance where possible.
- Evaluate different prediction horizons.
- Examine how predictions change across time periods.
- Document model limitations.

### December 3, 2026 — Final Analysis and Results

*Planned checkpoint*

Our goal is to finalize our model comparisons and determine whether economic indicators provide useful information for predicting recessions.

We will create visualizations showing model performance, important predictors, and predicted recession probabilities.

We will also begin organizing our findings for the final project report.

**Goals:**
- Finalize model comparisons.
- Create figures and tables.
- Summarize our main findings.
- Begin writing the final report.

### December 10, 2026 — Final Project

*Tentative target — confirm course deadline*

We plan to finish the report and make sure the project is reproducible.

We will document our data sources, methods, results, and limitations. Our final report will discuss how well machine learning can predict recessions and whether more complex models provide meaningful improvements over logistic regression.

## Data Sources

- FRED: https://fred.stlouisfed.org/
- NBER Business Cycle Dates: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- FRED Recession Indicator: https://fred.stlouisfed.org/series/USREC

## Tools

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

## Reproducibility

Our repository will contain the Python scripts used to collect, clean, and analyze the data, along with instructions for reproducing our results.

As we complete each checkpoint, we will update the corresponding section with our actual progress, results, and any changes to our original plan.

