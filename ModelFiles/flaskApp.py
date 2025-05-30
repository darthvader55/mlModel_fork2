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
        area = float(request.args.get('area'))

        # Debugging values
        print(f"Received values: area={area}")

        input_features = pd.DataFrame([[area]], columns=["area"])

        print(f"Input DataFrame:\n{input_features}")

        prediction = model.predict(input_features)
        prediction = np.round(prediction, 2)

        return f"The predicted house price is ${prediction[0]:,.2f}"

    except TypeError:
        return "Missing the required query parameters: area"

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "Invalid input values. Please ensure 'area' is numeric."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')                                        