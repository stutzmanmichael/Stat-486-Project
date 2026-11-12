Data and Reference Result
# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  
**Team:** Michael Stutzman and Ryan Morrow  
**Date:** 11/5/26

Use any prompts that help you describe your progress. Add a dated entry to the root `README.md` of your team's repository and link the relevant code and output there; this file is optional supporting material.

## Data

- Where did the data come from, and how did you obtain or prepare them?
The data is pulled directly from the FRED API, we pulled 71 tables in total. There are tables related to the following categories: yield curve and monetary policy, labor markets, output, households, housing and construction, and international markets. The data is on a monthly basis, not all tables have data for the full time frame (1-1-1959 to present).

- What data are available to your team now? What remains to be collected or cleaned?
The data needs relatively little cleaning, we need to look and investigate any outliers or unusual patterns in the data. 

- **Route C:** What does your first collected and cleaned sample contain? 
It has 71 variables timestamped for each month since 1959. There is missing data but there is not a reputable source to try and fill in the missing data. 

## Evaluation plan

- What comparison, split, metric, or diagnostic will help answer your question? Why?
The plan is to run a LASSO to eliminate any noisy predictors (why we chose so many variables to start with). Decision trees will then be used in the next step to create our prediction model and finalize how we “define” a recession based on the variables we have. 

- How will you avoid data leakage or another misleading comparison?
Because it is time series data, it would be best to train up until a certain year (probably around 2004, well before the Great Recession) and see how well our model can predict that. The hope is that it is an ongoing model that is able to predict a recession within 6 months, so the goal is to predict things that haven’t happened yet, not predict things in the past. 

- **Route B:** How will you validate locally before making a Kaggle submission? N/A

## First runnable result

- What baseline or other reference did you run? What did it produce?
We ran a lasso logistic regression model and compared it with a baseline that always predicts no recession. Results: 

| Metric | No-recession Baseline | Lasso |
|---|---|---|
| Accuracy | 98.55% | 57.25% |
| Precision | 0% | 3.28% |
| Recall | 0% | 100% |
| F-1 Score | 0 | 0.0635 |
| ROC AUC | 0.500 | 0.842 |

- Where are the code and output? Link them, and include the command needed to rerun the result if available.
The code for our data collection is in main.py, it can be run by inputting a FRED API key in the .env file after FRED_API_KEY=
API keys are publicly available but it is policy for each person to use their own and use a different one for every project so one is not included in the public github repo.  Also run uv sync.

- What did this first result suggest, and what will you try next?

-They suggest that economic indicators may contain useful information for predicting recession conditions. Our initial Lasso model did produce a lot of false positives though even if it did identify both positive observations. 

-High accuracy of the no recession-baseline also shows why accuracy alone can be misleading
Since there were only two positive observations in our test set we cannot and will not draw strong conclusions about how well the model would perform in other recessions

-Next we will investigate the false positives, try other indicators, and compare lasso with log. Reg. and random forests to see whether we can improve precision without sacrificing too much recall
