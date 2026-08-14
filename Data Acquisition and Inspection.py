import pandas as pd

# Load the dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv", na_values="?")

# Describing the dataset
print("Raw shape:", df.shape)
print(df.head(5))
print("\nInfo:")
df.info()