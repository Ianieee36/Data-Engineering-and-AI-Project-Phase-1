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