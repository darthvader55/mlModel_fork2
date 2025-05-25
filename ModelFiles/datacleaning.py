import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = read_csv("originaldata.csv")

# Show original data
print("Original Data:")
print(df.head())

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Dropping columns not needed eg. address and city
df = df.drop(columns=['Address', 'City', 'Date']) 

# Renaming "Price USD" and "Area (SQFT)" columns
df.rename(columns={'Price (USD)': 'price', 'Beds': 'bedrooms', 'Baths': 'baths', 'Area (SQFT)': 'area'}, inplace=True)

workspace = os.getenv('GITHUB_WORKSPACE')

# directory where datacleaning.py is (ModelFiles)
model_files_dir = os.path.join(workspace, 'ModelFiles')

# path for the output file
output_path = os.path.join(model_files_dir, 'cleaned_data.csv')

# Create the directory if it doesn't exist
os.makedirs(model_files_dir, exist_ok=True)

# Save cleaned data
df.to_csv(output_path, index=False)

print(output_path)
# Show cleaned data
print("Cleaned Data:")
print(df.head())

print("\nCleaned data saved to 'cleaned_data.csv'")