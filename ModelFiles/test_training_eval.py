import os
import pytest
import pandas as pd
from training_eval import load_cleaned_data, train_model
from sklearn.linear_model import LinearRegression

@pytest.fixture
def sample_cleaned_data_path():
    workspace = os.getenv('GITHUB_WORKSPACE')
    return os.path.join(workspace, 'ModelFiles', 'cleaned_data.csv')

def test_load_cleaned_data(sample_cleaned_data_path):
    df = load_cleaned_data(sample_cleaned_data_path)
    assert not df.empty, "Loaded DataFrame is empty."
    expected_columns = ['area', 'bedrooms', 'baths', 'price']
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

def test_train_model():
    # Sample dummy data for testing model training
    X_train = pd.DataFrame({
        'area': [1000, 1500, 2000],
        'bedrooms': [3, 4, 5],
        'baths': [2, 3, 4]
    })
    y_train = [250000, 350000, 450000]

    model = train_model(X_train, y_train)
    assert isinstance(model, LinearRegression), "Model is not an instance of LinearRegression."
    assert model.coef_ is not None, "Model coefficients are None."
