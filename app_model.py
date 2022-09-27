from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenid@ a la API predictora U0001F52E. Chequea las ventas que se alcanzarán este año según las inversiones en TV, radio y newspaper."

# 1. Crea un endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))
   
    def sql_query(query):
        connection = sqlite3.connect('advertising.db')
        cursor = connection.cursor()
        cursor.execute(query)
        ans = cursor.fetchall()
        names = [description[0] for description in cursor.description]

        return pd.DataFrame(ans,columns=names)

    df = sql_query('''SELECT * FROM datos''')
    
    X = df.drop(columns=['sales'])

    prediction = model.predict(X)

    return str(prediction)
    


# 2. Crea un endpoint que reentrene de nuevo el modelo con los datos disponibles en la carpeta data, que guarde ese modelo reentrenado, devolviendo en la respuesta la media del MAE de un cross validation con el nuevo modelo
# @app.route('/v1/retrain', methods=['PUT'])
# def retrain():
#     df = pd.read_csv('data/Advertising.csv', index_col=0)
#     X = df.drop(columns=['sales'])
#     y = df['sales']

#     model = pickle.load(open('data/advertising_model','rb'))
#     model.fit(X,y)
#     pickle.dump(model, open('data/advertising_model_v1','wb'))

#     scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

#     return "New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))


# app.run() ## NO SE USA PARA PYTHON ANYWHERE