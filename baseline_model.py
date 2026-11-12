import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Load collected data
df = pd.read_csv("fred_economic_data.csv")

# Rename month column to date
df = df.rename(columns={"month": "date"})

# Convert dates and sort chronologically
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Identify the beginning of each recession
recession_start = (
    (df["USREC"] == 1) &
    (df["USREC"].shift(1) == 0)
).astype(int)

# Target: recession begins within the next 6 months
df["target"] = pd.concat(
    [recession_start.shift(-i) for i in range(1, 7)],
    axis=1
).max(axis=1, skipna=False)

# Select initial predictors
features = [
    "T10Y3M",
    "UNRATE",
    "ICSA",
    "INDPRO",
    "HOUST",
    "PERMIT",
    "UMCSENT",
    "FEDFUNDS"
]

# Use a one-month lag as a preliminary
# precaution against publication delays
df[features] = df[features].shift(1)

# Remove observations without known outcomes
df = df.dropna(subset=["target", "USREC"])

# Only predict recession starts during expansions
df = df[df["USREC"] == 0]

# Chronological train/test split
train = df[df["date"] < "2007-07-01"]
test = df[df["date"] >= "2008-01-01"]

X_train = train[features]
y_train = train["target"].astype(int)

X_test = test[features]
y_test = test["target"].astype(int)

# Build model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# Fit model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Results
print("Training observations:", len(train))
print("Testing observations:", len(test))

print("\nTraining period:")
print(train["date"].min(), "to", train["date"].max())

print("\nTesting period:")
print(test["date"].min(), "to", test["date"].max())

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_prob))

print("\nAverage precision:")
print(average_precision_score(y_test, y_prob))

# Save predictions
results = test[["date", "target"]].copy()

results["predicted_probability"] = y_prob
results["predicted_class"] = y_pred

results.to_csv("baseline_predictions.csv", index=False)