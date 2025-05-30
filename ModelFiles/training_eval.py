import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_cleaned_data(path):
    df = read_csv(path)
    return df

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def save_model(model, output_path):
    dump(model, output_path)

def main():
    workspace = os.getenv('GITHUB_WORKSPACE')
    model_files_dir = os.path.join(workspace, 'ModelFiles')
    input_path = os.path.join(model_files_dir, 'cleaned_data.csv')
    output_model_path = os.path.join(model_files_dir, 'HousingModel.pkl')

    df = load_cleaned_data(input_path)

    X = df[["area", "bedrooms", "baths"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Model Now..")
    model = train_model(X_train, y_train)

    print("Evaluating Model Now...")
    r2 = evaluate_model(model, X_test, y_test)

    print(f"Model R² Score on Test Data: {r2:.4f}")

    save_model(model, output_model_path)
    print(f"Trained model saved to {output_model_path}")

if __name__ == "__main__":
    main()
