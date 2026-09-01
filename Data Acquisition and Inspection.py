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


# Binary-like check
binary_cols = [
    "IsDomainIP",
    "HasObfuscation",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "HasDescription",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo"
]

for col in binary_cols:
    unique_values = df[col].unique()
    print(f"\nColumn '{col}' unique values: {unique_values}")


