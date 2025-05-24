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
    area=request.args.get('area_sqft')
    prediction=model.predict([[float(area)]])
    prediction =np.round(prediction,2)
    return "The predicted house price is" + str(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
