from flask import Flask, request
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)
model = load("HousingModel.pkl")

@app.route('/')
def welcome():
    return "Welcome All To The House Price Predictor Webpage"

@app.route('/predict')
def predict_price():
    try:
        area = float(request.args.get('area'))
        bedrooms = float(request.args.get('bedrooms'))
        bathrooms = float(request.args.get('bathrooms'))

        print(f"Received values: area={area}, bedrooms={bedrooms}, bathrooms={bathrooms}")

        input_features = pd.DataFrame([[area, bedrooms, bathrooms]], columns=["area", "bedrooms", "bathrooms"])

        print(f"Input DataFrame:\n{input_features}")

        prediction = model.predict(input_features)
        prediction = np.round(prediction, 2)

        return f"The predicted house price is ${prediction[0]:,.2f}"

    except TypeError:
        return "Missing one or more required query parameters: area, bedrooms, bathrooms"

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "Invalid input values. Please ensure 'area', 'bedrooms', and 'bathrooms' are numeric."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
