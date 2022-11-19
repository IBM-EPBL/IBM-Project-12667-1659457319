"""
#Importing required libraries
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle
import requests


"""#Load the model and initialize Flask app"""

app = Flask(__name__)
filename = 'resale_model.sav'
model_rand = pickle.load(open(filename, 'rb'))

"""#Configure app.py to fetch the parameter values from the UI, and return the prediction"""


@app.route('/')
def index():
    return render_template('resaleintro.html')


@app.route('/predict')
def predict():
    return render_template('resalepredict.html')


@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    regyear = int(request.form.get('yearOfRegistration'))
    powerps = float(request.form.get('powerPS'))
    kms = float(request.form.get('kms'))
    regmonth = int(request.form.get('monthOfRegistration'))
    gearbox = request.form.get('gearbox')
    damage = request.form.get('notRepairedDamage')
    model = request.form.get('model')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuelType')
    vehicletype = request.form.get('vehicleType')
    new_row = {'yearOfRegistration': regyear, 'powerPS': powerps, 'kilometer': kms, 'monthOfRegistration': regmonth,
               'gearbox': gearbox, 'notRepairedDamage': damage, 'model': model, 'brand': brand, 'fuelType': fuelType, 'vehicleType': vehicletype}

    print(new_row)
    new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS',
                          'model', 'kilometer', 'monthofRegistration', 'fuelType', 'brand', 'notRepairedDamage'])
    new_df = new_df.append(new_row, ignore_index=True)
    labels = ['gearbox', 'notRepairedDamage',
              'model', 'brand', 'fuelType', 'vehicleType']
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'))
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i + '_labels'] = pd.Series(tr, index-new_df.index)
    labeled = new_df[['yearOfRegistration', 'powerPS',
                      'kilometer' 'monthOfRegistration']+[x+'_labels' for x in labels]]

    X = labeled.values

    y_prediction = predict(X)
    # print(y_prediction)
    return render_template('resalespredict.html', ypred=f'The resale value predicted is {y_prediction:.2f}')


def predict(X):
    # NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
    API_KEY = "JtQPfzFB5KgpVUWPiAstI6QMb6UkpOqhsZMARqEN6l-N"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
                                                                                     API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json',
              'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": ["vehicleType", "yearOfRegistration", "gearbox", "powerPS", "model", "kilometer", "monthOfRegistration", "fuelType", "brand", "notRepairedDamage"
                                                 ], "values": [X]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/6eba86d1-5b6b-40a7-8893-92fdbaeb7e9e/predictions?version=2022-11-16', json=payload_scoring,
                                     headers={'Authorization': 'Bearer ' + mltoken})
    # print("Scoring response")
    # print(response_scoring.json())

    return response_scoring.json()["predictions"][0]["values"][0][0]
    # output
    #{'predictions': [{'fields': ['prediction'], 'values': [[26544.185803974186]]}]}


"""#Run the app"""

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=False)
