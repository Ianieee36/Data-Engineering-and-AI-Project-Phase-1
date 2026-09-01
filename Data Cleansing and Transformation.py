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