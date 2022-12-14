# -*- coding: utf-8 -*-
"""flask.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1LtrfQqfRZpTMIIjeKKq0DizlENK0Tb7L
Build The Python Flask App
#Importing required libraries
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
from sklearn.preprocessing import LabelEncoder
import requests
"""#Load the model and initialize Flask app"""

app = Flask(__name__)
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "RtTfNjm2zENTuNTMOO2d6meo07NPaYavh9eIJvEZwMs3"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

"""#Configure python flask.py to fetch the parameter values from the UI, and return the prediction"""


@app.route('/')
def index():
    return render_template('resaleintro.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        regyear = int(request.form['regyear'])
        powerps = float(request.form['powerps'])
        kms = float(request.form['kms'])
        regmonth = int(request.form.get('regmonth'))
        gearbox = request.form['gearbox']
        damage = request.form['dam']
        model = request.form.get('modeltype')
        brand = request.form.get('brand')
        fuelType = request.form.get('fuel')
        vehicletype = request.form.get('vehicletype')
        new_row = {'yearOfRegistration': regyear, 'powerPS': powerps, 'kilometer': kms, 'monthOfRegistration': regmonth,
                   'gearbox': gearbox, 'notRepairedDamage': damage, 'model': model, 'brand': brand, 'fuelType': fuelType, 'vehicleType': vehicletype}

        print(new_row)
        new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS',
                              'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'brand', 'notRepairedDamage'])
        new_df = new_df.append(new_row, ignore_index=True)
        labels = ['gearbox', 'notRepairedDamage',
                  'model', 'brand', 'fuelType', 'vehicleType']
        mapper = {}
        for i in labels:
            mapper[i] = LabelEncoder()
            mapper[i].classes_ = np.load(str('classes'+i+'.npy'),allow_pickle=True)
            tr = mapper[i].fit_transform(new_df[i])
            new_df.loc[:, i + '_labels'] = pd.Series(tr, index=new_df.index)
        labeled = new_df[['yearOfRegistration', 'powerPS',
                          'kilometer','monthOfRegistration']+[x+'_labels' for x in labels]]

        X = labeled.values
        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"fields": [['vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS',
                              'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'brand', 'notRepairedDamage']], "values":X}]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/ab2cf516-865c-42d6-9e70-2b35dcdd9ad6/predictions?version=2022-11-18', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        predictions=response_scoring.json()
        predict_value=predictions['predictions'][0]['values'][0]
        print(predict_value)
        return render_template('resalepredict.html', ypred='The resale value predicted is {:.2f}$'.format(predict_value))
    else:
        return render_template('resalepredict.html')

"""#Run the app"""

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=False)
