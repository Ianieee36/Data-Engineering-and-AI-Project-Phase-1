import pandas as pd

# Load the dataset and make copy
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?")
df_original = df.copy(deep=True)


# removing duplicate urls
before_url = len(df)
df.drop_duplicates(subset="URL", inplace=True)

after_url= len(df)
print(f"Removed {before_url - after_url} rows with duplicate URL's. New shape: {df.shape}")


# checking duplicate domains
duplicates = df[df.duplicated(subset="Domain", keep=False)].sort_values("Domain")
print(duplicates)


# Identifier removal
df_clean = df.drop(columns=["FILENAME"])


# Handling outliers
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.boxplot(df["URLLength"].dropna())
plt.ylabel("URL Length")
plt.title("Distribution of URL Length")
plt.show()

# print values of URLLength that are considered outliers
p_outliers = (
    (df["URLLength"] > 3000) | (df["URLLength"] < 10)
)

print("Potential outliers in URLLength:")
print(df[p_outliers].loc[:, ["URL", "URLLength"]])


#TLD outliers
p_tld_outliers = (
    (df["TLDLength"] > 63)
)

print("Potential outliers in TLDLength:")
print(df[p_tld_outliers].loc[:, ["URL", "TLDLength"]])

plt.figure(figsize=(8, 5))
plt.boxplot(df["TLDLength"].dropna())
plt.ylabel("TLD Length")
plt.title("Distribution of TLD Length")
plt.show()


# numerical standardization
from sklearn.preprocessing import StandardScaler

numeric_columns = df_clean.select_dtypes(include=["number"]).columns # selecting only numeric columns
numeric_features = df_clean[numeric_columns].drop(columns=["label"]) # removing label column from features to be scaled

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

print("Scaled features:", scaled_features[:5])






### VISUALIZATIONS ###
import numpy as np
import seaborn as sns

target_col = "label"


numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target_col]

corr = df_clean[numeric_cols + [target_col]].corr()

# Correlation heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6}, linewidths=0.3)
plt.title("Correlation Heatmap - df_clean (Numeric Features & Target)")
plt.tight_layout()
plt.savefig("corr_heatmap_df_clean.png")
plt.close()

# Ranked feature-target correlations
target_corr = df_clean[numeric_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()

print("\nTOP 10 MOST NEGATIVELY CORRELATED WITH label")
print(target_corr.head(10).round(3).to_string())
print("\nTOP 10 MOST POSITIVELY CORRELATED WITH label")
print(target_corr.tail(10).round(3).to_string())

plt.figure(figsize=(9, 8))
top_corr = pd.concat([target_corr.head(8), target_corr.tail(8)])
colors = ["#d64550" if v < 0 else "#3a7d44" for v in top_corr.values]
plt.barh(top_corr.index, top_corr.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Target (label) - df_clean")
plt.xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig("target_corr_df_clean.png")
plt.close()

# signed pearson correlation heatmap for top 20 features most correlated with label
target_corr_all = df_clean[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
top20 = target_corr_all.abs().sort_values(ascending=False).head(20).index.tolist()
corr_small = df_clean[top20 + [target_col]].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr_small, cmap="coolwarm", center=0, square=True, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.7}, linewidths=0.4)
plt.title("Correlation Heatmap — Top 20 Features Most Correlated with label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("corr_heatmap_top20.png")


## Scatter plots
# LineOfCode vs NoOfExternalRef
sample = df_clean.sample(min(5000, len(df_clean)), random_state=42)

palette = {0: "#d64550", 1: "#3a7d44"}  # 0=Phishing, 1=Legitimate
labels = {0: "Phishing", 1: "Legitimate"}

plt.figure(figsize=(8, 6))
for val in [0, 1]:
    subset = sample[sample[target_col] == val]
    plt.scatter(subset["LineOfCode"], subset["NoOfExternalRef"], color=palette[val], label=labels[val], alpha=0.4, s=18)

plt.title("LineOfCode vs NoOfExternalRef (5,000-row sample)")
plt.xlabel("LineOfCode")
plt.ylabel("NoOfExternalRef")
plt.xlim(0, sample["LineOfCode"].quantile(0.98))
plt.ylim(0, sample["NoOfExternalRef"].quantile(0.98))
plt.legend(title="label")
plt.tight_layout()
plt.savefig("scatter_loc_vs_extref.png")
plt.close()