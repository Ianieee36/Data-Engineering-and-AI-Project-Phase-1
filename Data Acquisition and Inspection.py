import pandas as pd

# Load the dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?")

# Describing the dataset
print("Raw shape:", df.shape)
print(df.head(5))
print("\nInfo:")
df.info()


## Inspecting data

# Missing values
missing = df.isna().sum()
print("Columns containing missing values:")
print(missing[missing > 0].sort_values(ascending=False))

# Duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Important ranges
range_cols = ["URLSimilarityIndex"]
print("\nMinimum and maximum values:")
print(df[range_cols].agg(["min", "max"]))