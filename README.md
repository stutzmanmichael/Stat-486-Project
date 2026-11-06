# STAT 486 — Recession Prediction Project

**Team Members:** Michael Stutzman and Ryan Morrow 

## Project Overview

Our project investigates whether machine learning can predict economic recessions in the United States using historical economic data.

We plan to collect data from the Federal Reserve Economic Data (FRED) database and combine different economic indicators into a single dataset. We will then use machine learning models to identify patterns that might help predict future recessions.

**Research Question:** Can we use machine learning to predict recessions in the United States before they happen?

## Project Checkpoints

### October 27, 2026 — Project Proposal

For our project, we chose Route C, which involves building a dataset and using it in a reproducible machine learning pipeline.

We decided to use economic indicators from FRED, including unemployment rates, inflation, industrial production, and interest rates. We will also use historical recession dates from the National Bureau of Economic Research (NBER).

Our plan is to organize the data into monthly observations and compare different machine learning methods, including logistic regression, LASSO, and random forests.

We expect some challenges with missing data, different reporting frequencies, and the relatively small number of recessions in historical data.

We plan to evaluate our models using a time-based train/test split and metrics such as precision, recall, and F1 score.

### November 5, 2026 — Data and Reference Result

Our next step is to collect economic data from FRED and combine the different indicators into a monthly dataset.

We are starting with unemployment, industrial production, inflation, Treasury yield spreads, and federal funds rates. We also plan to use FRED's USREC series to identify historical recession periods.

For our initial reference model, we will use logistic regression to predict whether the economy will be in a recession three months into the future. We will compare this model against a simple baseline that predicts no recession.

We plan to use earlier years for training and later years for testing to avoid problems caused by randomly splitting time-series data.

Once the data pipeline and baseline are running, we will evaluate model performance and decide which additional features or models to explore.

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

## Future Work

We plan to compare logistic regression with LASSO and random forests, explore additional economic indicators, and examine which variables are most useful for predicting recessions.
