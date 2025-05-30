import pandas as pd
from datacleaning import clean_dataframe, rename_columns

def test_clean_dataframe():
    df = pd.DataFrame({
        'Address': ['A', 'B', 'B'],
        'City': ['X', 'Y', 'Y'],
        'Date': ['2020-01-01', '2020-01-02', '2020-01-02'],
        'Price (USD)': [100, 200, None],
        'Beds': [3, 2, 2],
        'Baths': [2, None, None],
        'Area (SQFT)': [1500, 1200, 1200]
    })
    cleaned_df = clean_dataframe(df)
    
    # Check no nulls
    assert cleaned_df.isnull().sum().sum() == 0

    # Check no duplicates
    assert len(cleaned_df) == len(cleaned_df.drop_duplicates())

    # Check required columns dropped
    for col in ['Address', 'City', 'Date']:
        assert col not in cleaned_df.columns

def test_rename_columns():
    df = pd.DataFrame({
        'Price (USD)': [100],
        'Beds': [3],
        'Baths': [2],
        'Area (SQFT)': [1500]
    })
    result = rename_columns(df)
    expected_columns = ['price', 'bedrooms', 'baths', 'area']
    assert all(col in result.columns for col in expected_columns)
