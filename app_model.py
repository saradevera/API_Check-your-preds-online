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
    my_model = pickle.load(open('data/advertising_model', 'rb')) # SE PUEDE METER EN LA MISMA LINEA

    # convertimos los datos de la llamada en test
    tv = request.args.get('tv', 0)
    radio = request.args.get('radio', 0)
    newspaper = request.args.get('newspaper', 0)

    if tv is 0 or radio is 0 or newspaper is 0:
        return 'Los datos introducidos no son correctos. Introduzca nuevos datos para realizar la prediccion'

    else:
        predictions = my_model.predict([[tv, radio, newspaper]])

        return str(predictions)
    
    


# 2. Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada. (/ingest_data)
@app.route('/ingest_data', methods=['POST'])
def ingest_data():

    tv = request.args.get('tv', 0)
    radio = request.args.get('radio', 0)
    newspaper = request.args.get('newspaper', 0)
    sales = request.args.get('sales', 0)

    connection = sqlite3.connect('advertising.db')
    cursor = connection.cursor()

    query = '''
    INSERT INTO datos
    (TV, radio, newspaper, sales)
    VALUES (?,?,?,?)
    '''

    cursor.execute(query, (tv, radio, newspaper, sales)).fetchall()
    connection.commit()


    return 'Has añadido los siguientes datos ' + str(tv) + ', ' + str(radio) + ', ' + str(newspaper) + ', ' + str(sales)


# 2b. Imprimir la base de datos por pantalla para comprobar el ejercicio anterior
@app.route('/print_db', methods=['GET'])
def print_db():

    connection = sqlite3.connect('advertising.db')
    cursor = connection.cursor()

    query = '''
    SELECT * FROM datos
    '''

    result = cursor.execute(query).fetchall()
    connection.commit()


    return jsonify(result)

# 3. Posibilidad de reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/retrain)
@app.route('/retrain', methods=['GET'])
def retrain():

    def sql_query(query):
        connection = sqlite3.connect('advertising.db')
        cursor = connection.cursor()
        cursor.execute(query)
        ans = cursor.fetchall()

        names = [description[0] for description in cursor.description]
        connection.commit()

        return pd.DataFrame(ans,columns=names)

    df = sql_query('''SELECT * FROM datos''')

    X = df.drop(columns=['sales'])
    y = df['sales']

    model = pickle.load(open('data/advertising_model','rb'))
    model.fit(X,y)

    pickle.dump(model, open('advertising_model_retrain','wb'))

    return 'Has reentrenado el modelo con últimos datos añadidos. \n\nCaracterísticas del modelo: ' + str(model)

