"""Check how changing the prediction cutoff affects our November 12 model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# These predictions were saved by baseline_model.py.
predictions = pd.read_csv("baseline_predictions.csv", parse_dates=["date"])
y_true = predictions["target"].astype(int)
probabilities = predictions["predicted_probability"]

# These are exploratory cutoffs, not tuned on a separate validation set.
thresholds = [0.50, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
rows = []

for threshold in thresholds:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rows.append({
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    })

results = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
results.to_csv("results/nov19_threshold_results.csv", index=False)

# Look at which months were incorrectly flagged at the original cutoff.
false_alarms = predictions[
    (predictions["target"] == 0) & (predictions["predicted_class"] == 1)
].copy()
false_alarms.to_csv("results/nov19_false_alarms.csv", index=False)

print(results.to_string(index=False))
print("\nFalse alarms by year at threshold 0.50:")
print(false_alarms.groupby(false_alarms["date"].dt.year).size().to_string())

# Plot how many positive months we catch versus how many false alarms occur.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results["false_positives"], results["true_positives"], "o-")
for _, row in results.iterrows():
    offset = (4, 5)
    if row["threshold"] == 0.10:
        offset = (-5, -18)
    elif row["threshold"] == 0.05:
        offset = (6, 12)
    ax.annotate(f'{row["threshold"]:g}',
                (row["false_positives"], row["true_positives"]),
                xytext=offset, textcoords="offset points")
ax.set_xlabel("False warnings")
ax.set_ylabel("Positive months correctly identified (out of 6)")
ax.set_title("Recession onset predictions: threshold tradeoff")
ax.set_ylim(-0.5, 6.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("results/nov19_threshold_tradeoff.png", dpi=160)
plt.close(fig)

print("\nSaved the threshold table, false alarms, and plot in results/.")
