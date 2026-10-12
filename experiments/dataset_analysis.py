import pandas as pd
df = pd.read_csv("data/resume_data.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nMatched score statistics:")
print(df["matched_score"].describe())

print("\nNumber of unique matched scores:")
print(df["matched_score"].nunique())