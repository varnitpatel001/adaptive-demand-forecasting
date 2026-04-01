import pandas as pd

# Load Kaggle dataset
df = pd.read_csv("data/train.csv")

# Step 1: Use only one store (VERY IMPORTANT)
df = df[df['Store'] == 1]

# Step 2: Select only required columns
df = df[['Date', 'Sales', 'Promo']]

# Step 3: Rename columns (standard format)
df.columns = ['date', 'sales', 'promo']

# Step 4: Sort by date
df = df.sort_values('date')

# Step 5: Save cleaned dataset
df.to_csv("data/raw.csv", index=False)

print("Cleaned dataset saved as data/raw.csv")
