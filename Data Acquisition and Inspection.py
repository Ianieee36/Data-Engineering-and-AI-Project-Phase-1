import pandas as pd

# Load the dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Describing the dataset
print(df.shape)
print(df.head())