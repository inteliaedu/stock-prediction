# Tratamiento de datos
import pandas as pd
import math
import numpy as np

# Tratamiento de fechas
from datetime import datetime, timedelta

# Visualización de datos
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# API library
from alpha_vantage.timeseries import TimeSeries 

# Normalización de datos
from sklearn.preprocessing import StandardScaler

def descargar_datos(config, plot=False):
    ts = TimeSeries(key=config["alpha_vantage"]["key"]) # Configuración básica para acceder a la API
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"]) 
    # Extraemos los datos de la empresa deseada y el tamaño de la extracción
    data_date = [date for date in data.keys()] # Creamos una lista de las fechas que se extraen
    data_date.reverse() # Ordenamos las fechas de menor a mayor

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]  # Extracción del cierre en los datos
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points:", num_data_points, display_date_range)

    if plot: # Visualización de datos en intervalos
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
        xticks = [data_date[i] if ((i%int(config["plots"]["xticks_interval"])==0 and (num_data_points-i) > int(config["plots"]["xticks_interval"])) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
        plt.grid(visible=None, which='major', axis='y', linestyle='--')
        plt.show()

    return data_date, data_close_price, num_data_points, display_date_range

def obtener_data_pandas(fechas, valores, corte):
    df = pd.DataFrame([fechas, valores], index=['Fecha', 'Cierre']).T
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df[df['Fecha']>=corte].reset_index(drop=True)
    # Creamos un dataframe con solo la columna cierre
    data_target = df.filter(['Cierre'])
    # Converitmos la columna cierre a un arreglo de numpy para el modelo
    target = data_target.values
    return data_target,target

def split_train_val_test(array, test_size, val_size, window_size):
    training_data_len = len(array) - test_size - val_size
    train_data = array[:training_data_len, :]
    val_data = array[training_data_len- window_size:training_data_len+val_size, :]
    test_data = array[training_data_len+val_size-window_size:, :]
    
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    sc_train = StandardScaler()
    train_data = sc_train.fit_transform(train_data)
    val_data = sc_train.transform(val_data)
    test_data = sc_train.transform(test_data)
    
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])
    for i in range(window_size, len(val_data)):
        X_val.append(val_data[i-window_size:i, 0])
        y_val.append(val_data[i, 0])
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    print('Número de registros y columnas (training):', X_train.shape)
    print('Número de registros y columnas (validation):', X_val.shape)
    print('Número de registros y columnas (test):', X_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, sc_train

def plot_predictions(config,predicted_stock_price,X_train,fechas, data_target,split_name):
    ultima_fecha = datetime.strptime(fechas[-1:][0], '%Y-%m-%d')
    date_after = ultima_fecha+ timedelta(days=1)
    fechas.append(str(date_after.strftime('%Y-%m-%d')))
    lista_fechas = fechas[-int(config["data"]["test_split_size"]):]
    training_data_len = X_train.shape[0] +  int(config["data"]["window_size"]) +  int(config["data"]["validation_split_size"]) + 1
    train = data_target[1100:training_data_len]
    valid = data_target.iloc[training_data_len:,:].copy()
    valid.loc[max(valid.index) + 1] =np.NaN
    valid['Predictions'] = predicted_stock_price
    RMSE = round(((valid.Cierre - valid.Predictions) ** 2).mean() ** .5,2)
    if split_name == "Full":
        # Visualising the results
        plt.figure(figsize=(10,5))
        plt.title('Modelo LTSM predicción Full '+config["alpha_vantage"]["symbol"]+' RMSE='+str(RMSE) )
        plt.xlabel('Fecha', fontsize=8)
        plt.ylabel('Precio Cierre USD ($)', fontsize=12)
        plt.plot(train['Cierre'])
        plt.plot(valid[['Cierre', 'Predictions']])
        plt.legend(['Train', 'Test', 'Predictions'], loc='upper left')
        plt.show()
    if split_name == "Pred":
        valid['Fecha'] = lista_fechas
        valid = valid.reset_index(drop=True)
        # Visualising the results
        plt.figure(figsize=(10,5))
        plt.title('Modelo LTSM predicción '+config["alpha_vantage"]["symbol"]+' RMSE='+str(RMSE) )
        plt.xlabel('Fecha', fontsize=8)
        plt.ylabel('Precio Cierre USD ($)', fontsize=12)
        plt.xticks(range(len(valid['Fecha'])), valid['Fecha'], rotation=45)
        plt.plot(valid[['Cierre', 'Predictions']])
        plt.legend([ 'Test', 'Predictions'], loc='upper right')
        plt.show()
        return RMSE, lista_fechas