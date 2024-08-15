import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
#import pickle
import joblib
import math
import base64
from flask import jsonify

app= Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, Welcome World!</p>"


@app.route('/predict/<ID>', methods = ['GET'])
def predict(ID):
    '''
    For rendering results on HTML GUI
    '''
    
    url_df = 'https://raw.githubusercontent.com/JEFFNASA/P7_FINAL_OCR/main/API/df_API_BIS.csv'
    df_test = pd.read_csv(url_df)
    #df_test = pd.read_csv('df_API_BIS.csv', encoding='utf-8')
    df_test.drop("TARGET", inplace=True, axis=1)
    liste_clients = list(df_test['SK_ID_CURR'].unique())
    
    probability_default_payment = np.array([0])
    model = joblib.load('best_model.joblib')
    seuil = 0.51
    ID = int(ID)
    if ID not in liste_clients:
        prediction="Ce client n'est pas répertorié"
    else :
        X = df_test[df_test['SK_ID_CURR'] == ID]#[df_test.columns.tolist()[1:]]
        X.drop('SK_ID_CURR', axis=1, inplace=True) 
        
        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"

    return jsonify({"prediction": prediction, "score": probability_default_payment.tolist()})

if __name__ == "__main__":
    app.run(debug=True)

