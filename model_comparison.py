"""Compare models on the same six-month recession-onset problem."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Use the exact same dataset, target, and split as baseline_model.py.
file = Path("fred_economic_data.csv")
if not file.exists():
    raise FileNotFoundError(
        "fred_economic_data.csv is missing. Run `uv run python main.py` "
        "with your FRED API key, or use your existing CSV."
    )

df = pd.read_csv(file)
df = df.rename(columns={"month": "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

recession_start = ((df["USREC"] == 1) & (df["USREC"].shift(1) == 0)).astype(int)
df["target"] = pd.concat(
    [recession_start.shift(-i) for i in range(1, 7)], axis=1
).max(axis=1, skipna=False)

features = ["T10Y3M", "UNRATE", "ICSA", "INDPRO",
            "HOUST", "PERMIT", "UMCSENT", "FEDFUNDS"]
df[features] = df[features].shift(1)
df = df.dropna(subset=["target", "USREC"])
df = df[df["USREC"] == 0]

train = df[df["date"] < "2007-07-01"]
test = df[df["date"] >= "2008-01-01"]
X_train, y_train = train[features], train["target"].astype(int)
X_test, y_test = test[features], test["target"].astype(int)

# Imputation and scaling are fit on training data only.
models = {
    "Always no recession": DummyClassifier(strategy="constant", constant=0),
    "Logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]),
    "LASSO logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="liblinear", l1_ratio=1.0, C=1.0,
            class_weight="balanced", max_iter=2000, random_state=42
        )),
    ]),
    "Random forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ]),
}

Path("results").mkdir(exist_ok=True)
metrics_rows = []
prediction_rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()

    metrics_rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    })

    for date, actual, predicted_class, probability in zip(
        test["date"], y_test, predicted, probabilities
    ):
        prediction_rows.append({
            "model": name,
            "date": date.strftime("%Y-%m-%d"),
            "target": actual,
            "predicted_class": predicted_class,
            "predicted_probability": probability,
        })

    if name == "Random forest":
        importance = pd.DataFrame({
            "feature": features,
            "importance": model.named_steps["classifier"].feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv("results/nov19_rf_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance["feature"][::-1], importance["importance"][::-1])
        ax.set_xlabel("Feature importance")
        ax.set_title("Random forest feature importance")
        fig.tight_layout()
        fig.savefig("results/nov19_rf_importance.png", dpi=160)
        plt.close(fig)

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv("results/nov19_model_comparison.csv", index=False)
pd.DataFrame(prediction_rows).to_csv("results/nov19_model_predictions.csv", index=False)

print(f"Training: {len(train)} months; testing: {len(test)} months")
print(metrics.to_string(index=False))
print("\nResults saved in results/.")
