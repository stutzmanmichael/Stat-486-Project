

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Load the monthly data we downloaded from FRED
df = pd.read_csv("fred_economic_data.csv", parse_dates=["month"])
df = df.sort_values("month").set_index("month")

# starting with five economic indicators
# yearly changes for inflation and industrial production
features = pd.DataFrame(index=df.index)

features["unemployment"] = df["UNRATE"]
features["yield_spread"] = df["T10Y2Y"]
features["fed_funds_rate"] = df["FEDFUNDS"]

features["inflation_12m"] = (
    df["CPIAUCSL"].pct_change(periods=12, fill_method=None) * 100
)

features["production_growth_12m"] = (
    df["INDPRO"].pct_change(periods=12, fill_method=None) * 100
)

# Shift the predictors back one month to avoid using values
# from the same month we're making the prediction
# This doesn't completely account for reporting delays or revisions
features = features.shift(1)

# Our goal is to predict whether we'll be in a recession
# three months from now
features["recession_in_3_months"] = df["USREC"].shift(-3)

# Remove months that don't have all the values we need
clean = features.dropna().copy()
clean["recession_in_3_months"] = (
    clean["recession_in_3_months"].astype(int)
)

# Save our cleaned dataset
Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

clean.to_csv("data/cleaned_sample.csv", index_label="month")

predictors = [
    "unemployment",
    "yield_spread",
    "fed_funds_rate",
    "inflation_12m",
    "production_growth_12m"
]

# Train on older data and test on the more recent years
train = clean.loc[clean.index < "2015-01-01"]
test = clean.loc[clean.index >= "2015-01-01"]

if train.empty or test.empty or train["recession_in_3_months"].nunique() < 2:
    raise ValueError("Not enough training or test data.")

X_train = train[predictors]
y_train = train["recession_in_3_months"]

X_test = test[predictors]
y_test = test["recession_in_3_months"]

# Scale the variables and fit our LASSO logistic regression
# LASSO can shrink less useful coefficients down to zero
lasso = make_pipeline(
    StandardScaler(),
    LogisticRegression(
    l1_ratio=1.0,
    solver="liblinear",
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
    )
)

lasso.fit(X_train, y_train)

# Compare LASSO to a model that always predicts no recession
no_recession = DummyClassifier(
    strategy="constant",
    constant=0
)

no_recession.fit(X_train, y_train)

# See how both models performed on the test data
rows = []

for name, model in [
    ("Always no recession", no_recession),
    ("LASSO logistic regression", lasso)
]:
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    ).ravel()

    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(
            y_test, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicted, zero_division=0
        ),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() == 2 else float("nan")
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn
    })

# Save the model results
metrics = pd.DataFrame(rows)
metrics.to_csv("results/lasso_metrics.csv", index=False)

# Save the LASSO coefficients to see which variables matter
coefficients = pd.DataFrame({
    "feature": predictors,
    "coefficient": lasso.named_steps[
        "logisticregression"
    ].coef_[0]
})

coefficients.to_csv(
    "results/lasso_coefficients.csv", index=False
)

# Print everything we need for our checkpoint
print(f"Clean sample: {len(clean)} months")
print(f"Training: {len(train)} months")
print(f"Testing: {len(test)} months")

print("\nTest-set results:")
print(metrics.to_string(index=False))

print("\nLASSO coefficients:")
print(coefficients.to_string(index=False))

print("\nSaved cleaned data and model results.")
