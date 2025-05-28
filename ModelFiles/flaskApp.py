from flask import Flask, request
from joblib import load
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
model=load("HousingModel.pkl")

@app.route('/')
def welcome():
    return "Welcome Everyone To The House Price Predictor Webpage"

@app.route('/predict')
def predict_price():
    try:
        bedrooms = float(request.args.get('bedrooms'))
        baths = float(request.args.get('baths'))
        area = float(request.args.get('area'))

        # arranging same way model was trained on
        # input_features = np.array([[bedrooms, baths, area]])  

        # Debugging values
        print(f"Received values: bedrooms={bedrooms}, baths={baths}, area={area}")

        # creating a DataFrame matching the feature names used in training
        input_features = pd.DataFrame([[bedrooms, baths, area]], columns=["bedrooms", "baths", "area"])

        print(f"Input DataFrame:\n{input_features}")

        prediction = model.predict(input_features)
        prediction = np.round(prediction, 2)

        return f"The predicted house price is ${prediction[0]:,.2f}"

    except TypeError:
        return "Missing one or more required query parameters: bedrooms, baths, area"

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "Invalid input values. Please ensure bedrooms, baths and area are numeric."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
