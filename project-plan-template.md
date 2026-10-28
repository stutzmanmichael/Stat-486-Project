# STAT 486 Project Plan

**Due:** Tuesday, October 27, 2026, at 11:59 PM Mountain Time  
**Team:** Ryan Morrow and Michael Stutzman
**Repository:** (https://github.com/stutzmanmichael/Stat-486-Project)
**Date updated:** 10/27/26

Use these prompts to plan your investigation. Keep the answers focused; revise the plan when evidence or instructor feedback changes your direction. The Oct. 27 checkpoint still requires a new, dated project-plan entry in the root `README.md` of your team's GitHub repository. Link this document there if you use it; this document does not replace that entry.

## Project question and purpose

- What specific question do you want to investigate?
  Can we use machine learning to predict recessions in the US before they happen?
  
- Why does this question interest your team, and what might someone learn from answering it?
  Michael is an econ minor so he was interested in it, it can help people prepare before the financial impacts caused by recessions. 
  
- What claim or expectation could your evidence support, challenge, or leave uncertain?
There is a lot of uncertainty behind trying to predict recessions, usually when one happens, there are a lot of precautions taken so it doesn’t happen again. Our evidence could still support the claims of a weak economy and what actions the government might take.

- What does one observation represent in the data you expect to use?
We expect to use one month as an observation, some of the datasets may be more or less precise in their measurement periods. 

## Chosen route

Choose one route and explain why it fits your question and available time.

- [ ] **A. Develop something new** in an ML method or system
- [ ] **B. Kaggle competition** with two meaningfully different pipelines
- [X] **C. Build a dataset** and use it in a reproducible ML pipeline

**Why this route:** [How does the route support the question and the team's intended contribution?]

## Data source and access

- What data will you use? Give the dataset, competition, API, or source name and a direct link if available.
https://fred.stlouisfed.org/
FRED has a lot of datasets about all different kinds of economic metrics (unemployment rate, industrial production, CPI, federal funds interest rate, etc.)
https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER has the official dates for past recessions

- What does each row or record represent, and which fields are relevant to the question?
Each row is a month, part of the goal is to figure out which metrics are useful in predicting recessions, this can help infer causation.

- How will the team obtain the data? Note access requirements, collection dates or versions, and any expected size or compute needs.
It is publicly available through an API, NBER might need a scraping tool. 

- What license, competition rules, privacy, ethical, or other use limits apply? Could any data be restricted or identifying?
No limits, free usage.

- What data quality issues might affect the investigation (for example, missingness, duplicates, labels, coverage, or changes over time)?
There are some changes in procedures that could lead to misleading results (change in M1 accounting in 2020 for example).

## Route-specific plan

Complete the subsection for your chosen route. For a different route, remove the unused subsections.

### Route C: New dataset and ML pipeline

- What useful dataset will you create, and what question will the ML analysis investigate?
It will be a dataset combining a lot of relevant economic characteristics, the analysis will compare differences between stable economic periods vs recession periods and can be used to see if the current state of the economy looks like it’s heading towards a recession.

- What source or sources will you combine or collect? Are collection and redistribution permitted?
From FRED, NBER, there might be other characteristics to look into such as dates of wars, technological advancements, etc. 

- What does one observation represent, and what will the dataset schema contain?
One observation is one month, the schema will contain the variables mentioned earlier and the variables that we create about predicting. (mostly floats, some dates)

- How will you collect, integrate, clean, or label the data? Which scripts will reproduce these steps?
Data will be collected through the API, cleaned and integrated in python. They will be labeled by date. 

- How will you record source locations, access or license terms, collection dates or versions, and transformations?
They will be recorded in comments in the code and on GitHub, there should be no issues with licensing or access, nor should there be changes in past data (only newer data appended on)

- What quality checks will you run (such as join coverage, duplicates, missingness, label consistency, or source changes)?
We will check for missing values, duplicates, incorrect joins, inconsistent dates, and label errors, and etc

- Can you preserve a permitted snapshot or small fixture? If raw data cannot be shared, how can an authorized reader rebuild or verify the cleaned data?
We can save a cleaned snapshot of the dataset so results can be reproduced even if FRED updates later

- What reference ML method will use the cleaned dataset, and what evidence will show both dataset fitness and ML usefulness?
We will use methods such as logistic regression, LASSO, random forests. Model performance and validation results will show whether the data is useful or not in predicting the recessions.

## Initial reference and proposed contribution

- What is the simplest credible baseline, reference method, or comparison for your question?
Start with a simple logistic regression using a few major economic indicators

- What will your team design, build, collect, or investigate beyond that reference?
Compare it with LASSO and random forest models using a larger set of FRED variables

- Which choices, code, data, papers, notebooks, packages, or pretrained models will come from others, and how will you credit them?
FRED data and any outside code, packages, or papers will be cited in the report and code comments

## Evaluation and evidence

- What evaluation design fits the question and how the data were collected (for example, a split, cross-validation, control, stability check, or data-quality diagnostic)?
Use a time-based train/test split- model trained on earlier years and tested on later- maybe log regression

- What metric or diagnostic will you use, and why does it match the question?
LSTM(long short term memory), recall, precision, random forests

- How will you compare your work with a baseline or other reference? What would count as evidence against your initial expectation?
Compare the model to a simple model that always predicts no recession or to a basic logistic regression

- What informative error, failure, uncertainty, tradeoff, or limitation will you inspect?
Pay attention to false negatives, where the model misses an actual recession

- What figures, tables, examples, or other outputs could make the results understandable?
ROC curve, Precision, recall, F1 score, 

## Risks and backup scope

- What could prevent the plan from working (for example, access, collection, compute, time, data quality, or competition constraints)?
FRED variables might have missing values, different reporting frequencies, or limited historical coverage. Recessions also aren't too common so maybe it'd be harder for a model to learn useful patterns. 

- What smaller version would still let you answer a useful question if a risk occurs?
Use fewer major indicators(unemployment, interest rates, inflation, industrial production, CPI, etc) and shorter time period with complete data

- What evidence or feedback would lead you to pivot, and how would you document the reason?
If the data quality is poor or the model does not beset a simple baseline we would document what changed and why.

## Reproducibility and team contributions

- Where will code, configuration, data instructions, and results live in the repository?
Code, data instructions, and results will be organized in Github with clean folders, README, and all the works

- What is the first small runnable result you aim to produce, and what will a reader need to reproduce it?
A cleaned FRED dataset and simple logistic regression that can be reproduced from the provided code and package requirements

- How will team members contribute substantively and review one another's work?
We will split up the work evenly and we are roommates so there is no running/hiding.

- What support, approval, or early instructor feedback would help you proceed?
If this isn't feasible we'd like to know early on(we know the question isn't too small but it might be too big?) Patience and kindness always! We’ll return it too :)

## Sources

List initial data sources, papers, competition pages, public notebooks, software, and other materials relevant to the plan. Include links and access or license details when known.

https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions

https://fred.stlouisfed.org/


