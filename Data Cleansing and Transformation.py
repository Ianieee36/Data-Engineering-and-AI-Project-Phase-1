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