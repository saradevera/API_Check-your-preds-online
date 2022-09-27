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
    return "Bienvenid@ a la API predictora \U0001F52E. Chequea las ventas que se alcanzarán este año según las inversiones en TV, radio y newspaper."

# 1. Ofrezca la predicción de ventas a partir de todos los valores de gastos en publicidad. (/predict)
@app.route('/predict', methods=['GET'])
def predict():
    # cargamos el modelo
    my_model = pickle.load(open('data.advertising_model', 'rb')) # SE PUEDE METER EN LA MISMA LINEA

    # convertimos los datos de la llamada en test
    tv = request.args.get('tv', 0)
    radio = request.args.get('radio', 0)
    newspaper = request.args.get('newspaper', 0)

    if tv is 0 or radio is 0 or newspaper is 0:
        return 'Los datos introducidos no son correctos. Introduzca nuevos datos para realizar la prediccion'

    else:
        predictions = my_model.predict([[tv, radio, newspaper]])

    # return de predictions
    return str(predictions)
    


# 2. Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada. (/ingest_data)
@app.route('/ingest_data', methods=['POST'])
def ingest_data():

    connection = sqlite3.connect('advertising.db')
    cursor = connection.cursor()

    query = '''
    INSERT INTO datos
    (TV, radio, newspaper, sales)
    VALUES (?,?,?,?)
    '''
    tv = request.args.get('tv', 0)
    radio = request.args.get('radio', 0)
    newspaper = request.args.get('newspaper', 0)
    sales = request.args.get('sales', 0)

    cursor.execute(query, (tv, radio, newspaper, sales)).fetchall()


# app.run() ## NO SE USA PARA PYTHON_ANYWHERE