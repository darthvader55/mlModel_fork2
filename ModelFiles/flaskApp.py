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
        baths = float(request.args.get('baths'))

        print(f"Received values: area={area}, bedrooms={bedrooms}, baths={baths}")

        input_features = pd.DataFrame([[area, bedrooms, baths]], columns=["area", "bedrooms", "baths"])

        print(f"Input DataFrame:\n{input_features}")

        prediction = model.predict(input_features)
        prediction = np.round(prediction, 2)

        return f"The predicted house price is ${prediction[0]:,.2f}"

    except TypeError:
        return "Missing one or more required query parameters: area, bedrooms, baths"

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "Invalid input values. Please ensure 'area', 'bedrooms', and 'baths' are numeric."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
