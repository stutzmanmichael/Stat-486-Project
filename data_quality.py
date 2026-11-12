import pandas as pd

df = pd.read_csv("fred_economic_data.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five observations:")
print(df.head())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMissing values by column (%):")
print(
    (df.isna().mean() * 100)
    .sort_values(ascending=False)
)

print("\nRecession indicator distribution:")
print(df["USREC"].value_counts())

print("\nSummary statistics:")
print(df.describe())