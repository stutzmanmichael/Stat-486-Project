# STAT 486 — Recession Prediction Project

**Team Members:** Michael Stutzman and Ryan Morrow

## Project Overview

Our project investigates whether machine learning can predict economic recessions in the United States using historical economic data from the Federal Reserve Economic Data (FRED) database.

We are combining economic indicators into a monthly dataset and comparing machine learning models to see whether these indicators can help predict future recession conditions.

**Research Question:** Can machine learning help predict U.S. recessions before they happen?

## Project Checkpoints

### October 27, 2026 — Project Proposal

We chose Route C, which involves building a dataset and using it in a reproducible machine learning pipeline.

Our plan was to collect economic indicators from FRED, including unemployment, inflation, industrial production, and interest rates, along with historical recession data based on NBER classifications.

We proposed comparing logistic regression, LASSO, and random forests using a time-based train/test split. We also identified missing data, different reporting frequencies, and the small number of historical recessions as potential challenges.

### November 5, 2026 — Data and Reference Result

#### Data

We successfully downloaded 71 economic indicators from FRED using Python and combined them into a monthly dataset with 815 observations and 72 columns, including the USREC recession indicator.

For our initial model, we selected five predictors: unemployment, the Treasury yield spread, the federal funds rate, annual inflation, and annual industrial production growth. We lagged these predictors by one month and created a target indicating whether the economy would be in a recession three months later.

After removing missing observations, we had 600 months of data. We still need to investigate historical coverage, reporting delays, and whether additional indicators would improve our predictions.

#### Evaluation Plan

We used a chronological train/test split with 462 months for training and 138 for testing. This allows us to evaluate predictions on later economic periods without randomly mixing past and future observations.

We compared LASSO logistic regression with a baseline that always predicts no recession. We used accuracy, precision, recall, F1 score, and ROC AUC to evaluate performance.

To reduce data leakage, we lagged our predictors and fitted the scaler using only the training data. However, historical FRED values may have been revised, which remains a limitation of our approach.

#### First Runnable Result

Our initial LASSO logistic regression produced the following results:

| Metric | No-Recession Baseline | LASSO |
|---|---|---|
| Accuracy | 98.55% | 57.25% |
| Precision | 0% | 3.28% |
| Recall | 0% | 100% |
| F1 Score | 0 | 0.0635 |
| ROC AUC | 0.500 | 0.842 |

LASSO correctly identified both positive recession observations in the test set but also produced 59 false positives. Although the model achieved 100% recall, there were only two positive observations, so we cannot draw strong conclusions yet.

The results suggest that some economic indicators may be useful for distinguishing recession conditions, but the model currently produces too many false alarms.

**Code and Results:**

- [FRED data collection](main.py)
- [LASSO model](lasso_checkpoint.py)
- [Cleaned dataset](data/cleaned_sample.csv)
- [Model performance](results/lasso_metrics.csv)
- [LASSO coefficients](results/lasso_coefficients.csv)

To reproduce the results, create a local `.env` file containing a valid `FRED_API_KEY`, then run:

```bash
uv sync
uv run python main.py
uv run python lasso.py
```

### November 12, 2026 — Working Investigation and Initial Results

**Data Collection and Pipeline**

We developed a reproducible Python pipeline that collects historical economic indicators from the [FRED database](https://fred.stlouisfed.org/), converts them to monthly observations, and integrates them into a single dataset. Each observation represents one month of U.S. economic activity.

The dataset includes indicators related to interest rates, employment, inflation, housing, consumer spending, and financial markets. The historical recession indicator (`USREC`) is used to construct a target identifying whether a recession begins within the next six months.

**Code and Reproduction:**
- [Data collection and integration](main.py)
- [Environment configuration](pyproject.toml)
- [Data quality checks](data_quality.py)
- [Baseline model](baseline_model.py)

Historical observations begin in January 1959, where available. FRED data may be revised, and API access requires a key.

**Data Quality**

| Quality Check | Result |
|---|---|
| Number of observations | 815 |
| Number of economic predictors | 71 |
| Duplicate rows | 0 |
| Non-recession months | 719 |
| Recession months | 95 |
| Missing recession labels | 1 |
| Missingness across predictors | Varies substantially |

[Full data quality results](data_quality_results.txt)

The quality checks revealed substantial differences in historical coverage. For example, the high-yield corporate bond spread has 95.5% missing values, while the federal funds rate has only 0.12% missing values. These differences may require us to exclude certain predictors or restrict the study period.

**Initial Model Results**

We trained a logistic regression model using eight economic indicators from the cleaned dataset. We used a chronological split with 507 training observations from January 1959 through June 2007 and 201 testing observations from July 2009 through May 2026. Missing values were handled using median imputation, and predictors were standardized before fitting.

| Metric | Result |
|---|---:|
| Accuracy | 87.6% |
| Precision | 0.0% |
| Recall | 0.0% |
| F1 Score | 0.0 |
| ROC AUC | 0.7393 |
| Average Precision | 0.0605 |
| False Positives | 19 |
| False Negatives | 6 |
| True Positives | 0 |

[Model results](baseline_results.txt) | [Model predictions](baseline_predictions.csv)

Although the model achieved an ROC AUC of 0.7393, it failed to correctly identify any upcoming recession at the default classification threshold. It produced 19 false warnings and missed all six positive observations. Its relatively high accuracy is misleading because a model that always predicted no upcoming recession would achieve approximately 97% accuracy.

These results suggest that the model has some ability to distinguish between higher and lower-risk periods, but its current classification performance is bad. Additionally, the testing period contains only one recession onset, limiting our ability to assess generalizability.

**Changes and Next Steps**

Our research question remains unchanged. However, the initial results emphasize the need to investigate class imbalance, feature transformations, and classification thresholds before introducing more complex models (decision trees/ random forests).

We plan to compare LASSO with standard logistic regression, investigate false positives, and explore whether changes in economic indicators are more predictive than their raw values.

Our main limitations are the small number of historical recessions, inconsistent predictor coverage, and potential look-ahead bias from data revisions and publication delays. We may narrow our feature set to indicators with longer historical coverage and evaluate performance across additional historical recession periods. One challenge with predicting recessions is that after one happens, there is usually a lot of safety nets put up to make sure a recession doesn't happen in the same way again. Our model needs to be more advanced to be able to predict recessions because they don't happen often and generally they don't follow a strict pattern in how they start. 

Our next question is whether feature engineering and regularization can improve recession prediction while reducing false warnings. One thing we need to keep in mind is not to regularize our model so much that it eliminates all false positives (because it would probably reduce the sensitivity to true positives as well). One challenge in this dataset is that the data does not go back all the way to the Great Depression, one of the strongest and most famous recessions (or depression) in the United States, which would be really good to train on.

### November 19, 2026 — Model Comparison

*Planned checkpoint*

We plan to introduce a random forest model and compare its performance with our logistic regression models.

We will focus on whether more complex methods improve recession predictions and examine which economic indicators contribute most to each model.

**Goals:**
- Train and evaluate a random forest.
- Compare model performance using the same evaluation period.
- Examine feature importance and classification errors.

### November 26, 2026 — Model Limitations and Refinement

*Planned checkpoint*

We plan to examine how consistently our models perform across different economic periods and whether changing the prediction horizon affects the results.

We will also consider the effects of historical data revisions, reporting delays, and the limited number of recession observations.

**Goals:**
- Evaluate model stability across time.
- Compare different prediction horizons.
- Document important limitations and potential improvements.

### December 3, 2026 — Final Analysis and Results

*Planned checkpoint*

We plan to finalize our model comparisons, create visualizations, and summarize our findings.

Our main focus will be determining whether the models provide useful information beyond a simple baseline and explaining the tradeoffs between identifying recessions and producing false alarms.

**Goals:**
- Finalize model evaluations.
- Create tables and visualizations.
- Summarize findings and begin the final report.

### December 10, 2026 — Final Project

*Tentative target — confirm course deadline*

We plan to complete our report and make sure our analysis is reproducible.

The final report will discuss our methods, results, limitations, and whether more complex machine learning approaches meaningfully improve recession predictions.

## Data Sources

- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- [NBER Business Cycle Dates](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions)
- [FRED USREC Recession Indicator](https://fred.stlouisfed.org/series/USREC)

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, requests, and uv.

## Reproducibility

Our Python scripts collect and prepare the economic data, fit the models, and save the results. Each checkpoint documents our progress, findings, and any changes to the original plan.

The raw FRED dataset can be regenerated using `main.py` and a personal FRED API key.
