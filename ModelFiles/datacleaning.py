import os
import pandas as pd

def load_data(filepath):
    """Load data from CSV."""
    return pd.read_csv(filepath)

def clean_dataframe(df):
    """Remove nulls, duplicates, and drop unneeded columns."""
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=['Address', 'City', 'Date'])
    return df

def rename_columns(df):
    """Rename columns for consistency."""
    return df.rename(columns={
        'Price (USD)': 'price',
        'Beds': 'bedrooms',
        'Baths': 'baths',
        'Area (SQFT)': 'area'
    })

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    workspace = os.getenv('GITHUB_WORKSPACE', os.getcwd())
    model_files_dir = os.path.join(workspace, 'ModelFiles')
    input_path = os.path.join(workspace, 'originaldata.csv')
    output_path = os.path.join(model_files_dir, 'cleaned_data.csv')

    # Load original data
    df = load_data(input_path)
    print("Original Data:")
    print(df.head())

    # Data cleaning steps
    df = clean_dataframe(df)
    df = rename_columns(df)

    # Save cleaned data
    save_cleaned_data(df, output_path)

    print("\nCleaned Data:")
    print(df.head())
    print(f"\nCleaned data saved to '{output_path}'")

if __name__ == "__main__":
    main()
