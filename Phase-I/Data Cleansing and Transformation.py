"""
Task 3: Data Cleansing and Transformation
PhiUSIIL Phishing URL Dataset

Builds directly on the Task 2 inspection findings. Produces a cleaned,
transformed DataFrame ready for modelling, and prints a justification
for every decision made along the way.

"""
import re
import numpy as np
import pandas as pd

# Load the dataset and make copy of the original dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?") 
df_copy = df.copy(deep=True)  
print(f"\nLoaded dataset: {df_copy.shape[0]:,} rows x {df_copy.shape[1]} columns\n")

print("=" * 50)

# Addressing any missing values
print("\n--- 1. Missing Values ---")
missing = df_copy.isnull().sum().sum()
print(f"Total missing cells: {missing}")

if missing > 0:
    raise ValueError(
        "Missing values detected."
        "Add an explicit imputation strategy before proceeding."
    )
print("No action needed - dataset has no missing values.\n")

print("=" * 50)

# Addressing duplicated records
print("\n--- 2. Duplicated records ---")
before = len(df_copy)
dup_url_count = df_copy["URL"].duplicated().sum()
print(f"Duplicate URL values before dedup: {dup_url_count}")

# Drop duplicated records using copy
df_drop_dup = df_copy.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
print(f"Rows removed : {before - len(df_drop_dup)}")
print(f"Rows remaining : {len(df_drop_dup):,}\n")

print("=" * 50)

# Addressing invalid values
print("\n--- 3. Invalid values [IsHTTPS and IsDomainIP] ---")

# IsHTTPS: recompute from the actual URL schemed prefix, not the stored flag
before_https_mismatch = (
    (df_drop_dup["IsHTTPS"] == 1) & (~df_drop_dup["URL"].str.startswith("https://"))
).sum()
df_drop_dup["IsHTTPS"] = df_drop_dup["URL"].str.startswith("https://").astype(int)
print(f"IsHTTPS corrected on {before_https_mismatch} rows "
      f"(recomputed from URL scheme prefix). ")
 
# IsDomainIP: recompute from a strict IPv4 pattern match on Domain
ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
is_bare_ip = df_drop_dup["Domain"].astype(str).str.match(ip_pattern)
before_ip_mismatch = ((df_drop_dup["IsDomainIP"] == 1) & (~is_bare_ip)).sum()
df_drop_dup["IsDomainIP"] = is_bare_ip.astype(int)
print(f"IsDomainIP corrected on {before_ip_mismatch} rows "
      f"(recomputed from strict IPv4 pattern match on Domain).\n")

print("=" * 50)

# Addressing inconsistent categories and formatting issues
print("\n--- 4. Inconsistent categories & formatting ---")

# Title: the placeholder string "0" only appears when HasTitle=0 and does
# not represents a real page title, so it is cleared to an empty string
placeholder_count = (
    (df_drop_dup["HasTitle"] == 0) & (df_drop_dup["Title"] == "0")
).sum()
df_drop_dup.loc[
    (df_drop_dup["HasTitle"] == 0) & (df_drop_dup["Title"] == "0"), "Title"
] = np.nan
print(f"Title placeholder '0' cleared to NaN on {placeholder_count} rows.")

# Title: strip stray whitespace; TLD: standardise casing to lowercase
df_drop_dup["Title"] = df_drop_dup["Title"].astype(str).str.strip()
df_drop_dup["TLD"] = df_drop_dup["TLD"].astype(str).str.lower()
print("Title whitespace stripped; TLD standardised to lowercase.\n")

print("=" * 50)

# Fixing incorrect / inefficient data types
print("\n--- 5. Data type fixes ---")
binary_cols = [
    "IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle", "HasFavicon",
    "Robots", "IsResponsive", "HasDescription", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo",
    "label",
]
for col in binary_cols:
    df_drop_dup[col] = df_drop_dup[col].astype("int8")
df_drop_dup["TLD"] = df_drop_dup["TLD"].astype("category")
print(f"Cast {len(binary_cols)} binary columns to int8; TLD to category dtype.\n")
 
print("=" * 50)

# Investigating outliers 
print("\n--- 6. Outlier investigation ---")
skewed_count_cols = [
    "LineOfCode", "LargestLineLength", "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef", "URLLength",
]

print("IQR-based outlier scan (fraction of rows beyond 1.5*IQR fence):")
for col in skewed_count_cols:
    q1, q3 = df_drop_dup[col].quantile(0.25), df_drop_dup[col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    pct = ((df_drop_dup[col] > upper) | (df_drop_dup[col] < lower)).mean() * 100
    print(f"{col:20s} {pct:5.2f}% beyond fence (max={df_drop_dup[col].max():,.0f})")

# Cross-check: are these outliers noise, or do they carry class signal?
for col in skewed_count_cols:
    q1, q3 = df_drop_dup[col].quantile(0.25), df_drop_dup[col].quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    loc_outliers = df_drop_dup[df_drop_dup[col] > upper]
    print(f"\nLabel distribution of {col} outliers:")
    print(loc_outliers["label"].value_counts(normalize=True))
 
print("=" * 50)

# Applying Transformations
print("\n--- 7. Transformations ---")

# Log-transform the heavily right-skewed count/size columns
for col in skewed_count_cols:
    df_drop_dup[f"{col}_log"] = np.log1p(df_drop_dup[col])
print(f"Added log1p-transformed columns for: {skewed_count_cols}")

# TLD has 695 unique values - one-hot encoding would add ~695 columns.
# The dataset already provides TLDLegitimateProb, a derived feature
# summarising TLD trustworthiness, so raw TLD is excluded from the
# modelling matrix in favour of that existing numeric feature.
print("TLD retained in the full cleaned file, but excluded from the "
      "ML feature matrix in favour of TLDLegitimateProb (already "
      "derived, avoids a 695-category encoding).")
 
# Non-predictive identifier/free-text columns dropped from the ML matrix
non_predictive_cols = ["FILENAME", "URL", "Domain", "Title"]
print(f"Non-predictive columns dropped from the ML matrix: {non_predictive_cols}\n")
 
print("=" * 50)
 
# Building the final cleaned dataset and the ML-ready model matrix
print("\n--- Final outputs ---")
df_clean = df_drop_dup.copy(deep=True)
 
drop_cols = non_predictive_cols + ["TLD"] + skewed_count_cols  # keep *_log versions instead
df_model = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
 
print(f"Cleaned dataset (with traceability columns): "
      f"{df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
print(f"ML-ready model matrix: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns")

# Save both outputs to for Task 4
df_clean.to_csv("PhiUSIIL_cleaned.csv", index=False)
df_model.to_csv("PhiUSIIL_model_matrix.csv", index=False)
print("\nSaved: PhiUSIIL_cleaned.csv, PhiUSIIL_model_matrix.csv")