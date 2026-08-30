"""
Task 2: Data Acquisition, Inspection, and Documentation
PhiUSIIL Phishing URL Dataset
 
This script loads the dataset and produces every result referenced in the
Task 2 write-up: structure summary, dtype inspection, missing values,
duplicate records, value-range checks, unique-value checks, and the
inconsistency checks (Title/HasTitle mismatch, URLLength vs actual URL
string length, duplicate-URL label consistency).

"""


import pandas as pd
import re

# Data Loading
print("=" * 50)
print("1. Data Loading")
print("=" * 50)

df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?")   
print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

# Data Structure
print("=" * 50)
print("2. Data Structure")
print("=" * 50)

print(f"Number of records : {len(df):,}")
print(f"Number of features : {df.shape[1]} (55 predictors + 1 label)\n")

print("------------ Data Types -------------")
print(df.dtypes)
print("\n")

print("--- Label Distribution ---")
print(df["label"].value_counts())
print("\n")
print(df["label"].value_counts(normalize=True).rename("proportion"))

# Inspect for any missing and duplicate records
print("\n")
print("=" * 50)
print("3. Data Inspection")
print("=" * 50)

print("\n3.1 Missing Values")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values in any column.")
else:
    print(missing[missing > 0])

print("\n3.2 Duplicate Records")
print("Fully duplicated rows      :", df.duplicated().sum())
print("Duplicated URL values      :", df["URL"].duplicated().sum())
print("Duplicated FILENAME values :", df["FILENAME"].duplicated().sum())

# Check whether duplicate URLs carry conflicting labels
dup_urls = df[df.duplicated("URL", keep=False)]
label_conflict = dup_urls.groupby("URL")["label"].nunique()
conflicting = label_conflict[label_conflict > 1]
print(f"Rows involved in duplicate URLs : {len(dup_urls)}")
print(f"Duplicate URLs with CONFLICTING labels : {len(conflicting)}")


# Inspect value ranges
print("\n3.3 Value Ranges (numeric summary)")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.describe().T)

print("\n--- Binary/flag columns: confirm values are only {0, 1} ---")
binary_candidates = [
    "IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle", "HasFavicon",
    "Robots", "IsResponsive", "HasDescription", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo",
    "label",
]
for col in binary_candidates:
    print(f"{col:25s} -> {sorted(df[col].unique().tolist())}")
 
print("\n--- Ratio/probability columns: confirm within [0, 1] ---")
ratio_cols = [
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL",
    "SpacialCharRatioInURL",
]
for col in ratio_cols:
    out_of_range = df[(df[col] < 0) | (df[col] > 1)]
    print(f"{col:25s} min={df[col].min():.4f}  max={df[col].max():.4f}  "
            f"out_of_range={len(out_of_range)}")
 
print("\n--- Match/similarity score columns: confirm within [0, 100] ---")
for col in ["DomainTitleMatchScore", "URLTitleMatchScore", "URLSimilarityIndex"]:
    print(f"{col:25s} min={df[col].min():.4f}  max={df[col].max():.4f}")


# Inspect unique values
print("\n3.4 Unique Values (identifier / categorical columns)")

for col in ["FILENAME", "URL", "Domain", "TLD", "Title"]:
    print(f"{col:10s}: {df[col].nunique():,} unique / {len(df):,} rows")

print("\n--- Top 10 TLD values ---")
print(df["TLD"].value_counts().head(10))


#Inspect inconsistencies
print("\n3.5 Inconsistent / Invalid entry checks")

# 1. Title placeholder mismatch
print("------ Title Placeholder Mismatch ------")
empty_title = df["Title"].astype(str).str.strip() == ""
mismatch = df[(df["HasTitle"] == 0) & (-empty_title)]
print(f"\n[1] Rows with HasTitle=0 but non-blank Title text : {len(mismatch)}")
if len(mismatch) > 0:
    print("\nSample values in 'Title' for these rows:")
    print(mismatch["Title"].value_counts().head(5))
    print("     -> These are the placeholder string '0', not real titles.")

# 2. URLLength vs actual string length
print("\n------ URLLength vs actual string length ------")
diffs = df["URL"].str.len() - df["URLLength"]
print(f"\n[2] URL string length vs URLLength field:")
print(diffs.value_counts().head(10))
print("    -> Majority of rows are off by exactly 1; treated as a known")
print("       artefact of the original feature-extraction pipeline, not")
print("       something to silently recompute.")

# 3. DomainLength consistency
print("\n------ DomainLength Consistency ------")
mismatch_dlen = df[df["DomainLength"] != df["Domain"].astype(str).str.len()]
print(f"\n[3] Rows where DomainLength != len(Domain): {len(mismatch_dlen)}")
 
# 4. IsDomainIP sanity check (spot sample)
print("\n------ IsDomainIP sanity check ------")
 
ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
is_bare_ip = df["Domain"].astype(str).str.match(ip_pattern)
 
flagged = df[df["IsDomainIP"] == 1]
correct = flagged[is_bare_ip[flagged.index]]
mismatched = flagged[~is_bare_ip[flagged.index]]
 
print(f"\n[4] Rows flagged IsDomainIP=1     It       : {len(flagged)}")
print(f"    Confirmed bare IP address             : {len(correct)}")
print(f"    NOT a bare IP (mismatch)               : {len(mismatched)}")
 
if len(mismatched) > 0:
    print("\n    Sample of mismatched Domain values:")
    print(mismatched["Domain"].drop_duplicates().head(5).to_string(index=False))
 
missed = df[is_bare_ip & (df["IsDomainIP"] == 0)]
print(f"\n    Bare IP domains with IsDomainIP=0 (missed flags): {len(missed)}")
 

# 5. Check IsHTTPS consistency
print("\n------ IsHTTPS consistency  ------")
is_https_url = df["URL"].str.startswith("https://")
mismatch = df[(df["IsHTTPS"] == 1) & (-is_https_url)]

print(f"Total rows checked : {len(df):,}")
print(f"Consistent rows : {len(df) - len(mismatch):,}")
print(f"Inconsistent rows : {len(mismatch):,}")
print(f"Consistency rate : {(len(df) - len(mismatch)) / len(df):.4%}")
print(f"\n  IsHTTPS=1 but URL is NOT https://  : {len(mismatch):,}")
print(f"  IsHTTPS=0 but URL IS https://       : {(df['IsHTTPS'] == 0)[is_https_url].sum():,}")
print(f"  URL has no http(s) scheme at all    : {(~df['URL'].str.startswith(('http://','https://'))).sum():,}")
print("\n--- Label distribution of IsHTTPS=1 mismatches ---")
print(mismatch["label"].value_counts())





