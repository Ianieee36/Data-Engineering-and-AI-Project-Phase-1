import pandas as pd

# Load the dataset and make copy
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?")
df_original = df.copy(deep=True)