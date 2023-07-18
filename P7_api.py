# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 22:46:47 2023

@author: Paul
"""

from flask import Flask, jsonify, request
import pandas as pd
import dill as pickle
import numpy as np


app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def helloworld():
    if(request.method == 'GET'):
        num = request.args.get('number')
        data = {"data": "Hello World",
                "number": num}
        return jsonify(data)

@app.route('/predict', methods=['GET'])
def predict():
    username = int(request.args.get('customer'))
    threshold = 0.5757575757575758
    data = pd.read_csv('C:/Users/Paul/Documents/Cours Data Scientist OC/Projet 7/data/cleaned_data.csv', index_col=0)
    print(data.index)
    list_features = pickle.load(open('Pickles/features.pkl', 'rb'))  
    model = pickle.load(open('Pickles/randfor.pkl', 'rb')) # load trained model
    transformers = pickle.load(open('Pickles/transformers.pkl', 'rb')) #load transformers
    list_transformers = [i for i in transformers]  #list of transformers names (columns to be transformed)
    explainer = pickle.load(open('Pickles/explainer.pkl', 'rb')) #load lime explainer
    distributions = pickle.load(open('Pickles/distributions.pkl', 'rb'))
    print('Verifying ID: '+str(username))
    if username not in data.index:
        return {
            'Status': 'Error',
            'Message': 'Error: Unknown ID'
        }
    else:
        data_user = pd.DataFrame(data.loc[[username]])
        data_user = data_user.drop('TARGET', axis=1)
        for i in list_transformers:
            data_user[i] = transformers[i].transform(data_user[i])
        proba = model.predict_proba(data_user.values.reshape(1,-1))[0][1]
        prediction = 1 if proba > threshold else 0
        score = round(float(proba), 3)
        expl_details = explainer.explain_instance(
            data_user.values.reshape(-1),
            model.predict_proba,
            num_features=6
        )
        expl_details_map, expl_details_list = pd.DataFrame(expl_details.as_map()[1], columns=['Feature_idx', 'Scaled_value']), expl_details.as_list()
        names_main_features = []
        for i in expl_details_map['Feature_idx']:
            names_main_features.append(list_features['all_features'][i])

        feat_to_plot = [i for i in distributions if i in names_main_features]

        distributions_to_plot = {}
        for i in feat_to_plot:
            distributions_to_plot[i] = distributions[i]
        return {
                'Status': 'Success',
                'Prediction': int(prediction),
                'Score': score,
                'Threshold': round(threshold, 3),
                'User info': data_user.to_dict(),
                'Explainer map': expl_details_map.to_dict('list'),
                'Explainer list': expl_details_list,
                'Distributions': distributions_to_plot
            }


if __name__ == '__main__':
    app.run(debug=True,port=5000)