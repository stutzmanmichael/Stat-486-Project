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

We successfully downloaded 71 economic indicators from FRED using Python and combined them into a monthly dataset with 815 observations and 73 columns total (71 economic indicators, the USREC recession indicator, and the month column).

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
- [LASSO model](lasso.py)
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

### November 19, 2026 — Model Diagnostics and Comparison

**Evidence so far (Route C)**

Our dataset from FRED has 815 monthly observations, 71 economic indicators, and the USREC recession label. Our quality checks found no duplicate rows, but historical coverage varies a lot across variables. For example, the high-yield corporate bond spread has about 95.5% missing values, while the federal funds rate has about 0.12% missing values. This means we cannot just use all the variables without thinking about missing data and the time periods they cover.

We compared four models on the same problem: predicting whether a recession **starts in the next six months** using eight economic indicators. We used 507 older months for training and 201 later months for testing. We lagged predictors by one month and used training-set-only preprocessing to reduce leakage.

| Model | Accuracy | Precision | Recall | ROC AUC | Average precision | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always no recession | 97.0% | 0% | 0% | 0.500 | 0.0299 | 0 | 6 |
| Logistic regression | 87.6% | 0% | 0% | 0.739 | 0.0605 | 19 | 6 |
| LASSO logistic regression | 88.1% | 0% | 0% | 0.744 | 0.0616 | 18 | 6 |
| Random forest | 97.0% | 0% | 0% | 0.762 | 0.0662 | 0 | 6 |

The random forest had the best ROC AUC and average precision, but none of the models correctly classified any of the six positive test months at their default cutoffs. The random forest predicted no upcoming recessions at all, giving it the same accuracy as the no-recession baseline. This shows why accuracy is a bad metric to rely on by itself when recession onsets are so rare.

The [random forest feature importance results](results/nov19_rf_importance.csv) showed that the federal funds rate (31.2%) had the highest importance, followed by industrial production (15.3%), the unemployment rate (11.7%), and the 10-year minus 3-month Treasury yield spread (11.6%). These are useful predictors to investigate, although feature importance does not prove that any of them cause recessions. See the [feature importance figure](results/nov19_rf_importance.png).

We also used the logistic regression probabilities saved on November 12 to see what happens when we change the cutoff:

| Probability cutoff | Positive months found (out of 6) | False alarms | Precision | Recall |
|---|---:|---:|---:|---:|
| 0.50 | 0 | 19 | 0.0% | 0.0% |
| 0.01 | 2 | 52 | 3.7% | 33.3% |
| 0.005 | 5 | 58 | 7.9% | 83.3% |
| 0.001 | 6 | 67 | 8.2% | 100.0% |

The [false-alarm table](results/nov19_false_alarms.csv) also showed a pattern: of the 19 false positives from logistic regression at a 0.50 cutoff, 3 occurred in 2020, 9 in 2023, and 7 in 2024. All six actual positive test months were September 2019 through February 2020, ahead of the 2020 recession. The model issued more warnings after that recession than in the lead-up to it.

**Interpretation and next step**

One informative failure is that all four models missed every positive month at the default cutoff. Even the random forest, which ranked risk a little better based on ROC AUC, did not give a single positive warning. Lowering the logistic regression cutoff caught more positive months, but it also created many more false alarms. So we are learning that a model can have some useful ranking ability without yet being a useful early-warning system.

Our test set has only six positive months, all leading up to the same 2020 recession. That makes it hard to tell if these results would carry over to other recessions. We also used the test outcomes to explore cutoff choices, so the lower cutoffs are not validated improvements. Historical FRED revisions and publication delays could make real-world predictions harder too.

Next we want to use an earlier validation period to select a cutoff, then evaluate it on a later untouched test period. We also want to investigate why the federal funds rate has the highest importance and try changes in economic indicators instead of only their raw levels. More historical evaluation periods would help us see whether results hold up beyond the 2020 recession.

**Code, results, and figures:**

- [Data collection](main.py), [data-quality checks](data_quality.py), and [quality-check results](data_quality_results.txt)
- [Model comparison code](model_comparison.py) and [comparison table](results/nov19_model_comparison.csv)
- [Saved model predictions](results/nov19_model_predictions.csv)
- [Random forest importance table](results/nov19_rf_importance.csv) and [importance plot](results/nov19_rf_importance.png)
- [Threshold analysis code](threshold_analysis.py), [threshold results](results/nov19_threshold_results.csv), and [threshold tradeoff plot](results/nov19_threshold_tradeoff.png)
- [False alarm months](results/nov19_false_alarms.csv)

To reproduce these results, use the local FRED CSV generated with `main.py` and run:

```bash
uv sync
uv run python model_comparison.py
uv run python threshold_analysis.py
```

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
