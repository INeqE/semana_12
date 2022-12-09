import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.svm import SVC
from sklearn.metrics import classification_report

"""### Cargamos la data que utilizaremos, la de Microsoft (MSFT)"""

def app():

    st.title('Modelo SVC')
    ticker='MSFT'
    # st.subheader("Establecemos el año 2015")
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_dis = pd.read_csv(query_string)

    #Filtro por simbolo MSFT y obtención de dataframe
    st.subheader("Filtramos por simbolo MSFT y obtenemos el dataframe")
    df_dis['symbol']='MSFT'
    st.write(df_dis)

    # Creación de variables de predicción: Toma en cuenta precios de apertura (Open )y cierre (Close), precios pico (High) y bajo (Low)
    df_dis['Open-Close'] = df_dis.Open - df_dis.Close
    df_dis['High-Low'] = df_dis.High - df_dis.Low

    # Se guardan dichos valores en la variable X
    st.subheader("Guardamos los valores relevantes en la variable x")
    X = df_dis[['Open-Close', 'High-Low']]
    st.write(X.tail(4))

    #Haciendo la definición del objetivo {0} o {1}
    y = np.where(df_dis['Close'].shift(-1) > df_dis['Close'], 1, 0)

    st.subheader("Realizamos la predicción")
    from sklearn.model_selection import train_test_split
    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    #Entrenamiento o training del modelo, importando la libreria SVC
    #Indicar datos de entrenamiento (x_train y y_train)
    modelo = SVC().fit(x_train, y_train)

    #Haciendo predicción segun datos de testeo
    y_predict = modelo.predict(x_test)

    # Show clasiification report with formated cells
    report = classification_report(y_test, y_predict, output_dict=True)
    st.write(pd.DataFrame(report))


    st.subheader("Realizamos un test con datos ingresados")
    test = [[1.160004 , 2.430001],[-0.110001, 1.050004]]
    df = pd.DataFrame(test, columns=['Open-Close', 'High-Low'])
    y_predict = modelo.predict(df)
    st.write(y_predict)