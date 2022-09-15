
# MAIN LIBS
import os
import pandas as pd
import numpy as np
import pickle
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




# Check Working Directory
dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(dir_path)

print ("---Path check---")
print (dir)
print(os.getcwd())
print (dir_path+"/utils")

# sys.path is a list of absolute path strings
sys.path.append(dir_path+"/utils")

from functions_ml import *
from datetime import datetime

# SKLEARN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error

# DATA PROCESSING FUNCTION
def data_processing (df_fifa):
    """ Data Processing for the Dataframe"""
    # cleaning Weight Field
    field_cleanerx (df_fifa,"Weight","kg")

    # cleaning Height Field
    field_cleanerx (df_fifa,"Height","cm")

    # Create indexes with the remain strings which are Feet and Inches
    # Converting the inches and feet values in metric
    lista_inches = df_fifa[df_fifa['Height'].str.contains('\D+')==True]["Height"].index
    convert_inch_in_row (lista_inches,df_fifa,"Height")

    # Create indexes with the remain strings which are lbs
    lista_libs = df_fifa[df_fifa['Weight'].str.contains('\D+')==True]["Weight"].index
    convert_lbs_to_kg (lista_libs,df_fifa,"Weight")

    # We apply the conversion to all money related columns
    df_fifa["Value"] = df_fifa["Value"].apply(convert_monetary)
    df_fifa["Wage"] = df_fifa["Wage"].apply(convert_monetary)
    df_fifa["Release Clause"] = df_fifa["Release Clause"].apply(convert_monetary)

    # We apply to force all converted columns so far to float
    listnames = ["Height","Weight","Value","Wage","Release Clause"]
    force_col_convert(listnames,df_fifa,float)

    # Conversion of the Joined column to datetime
    df_fifa["Joined"] = pd.to_datetime(df_fifa["Joined"])

    # Check for Star string and remove
    # for i in df_fifa.columns:
    #     try:
    #         if df_fifa[i].dtype == 'object':
    #             df_fifa[i] = df_fifa[i].apply(lambda x : x.replace("★","")).astype(float)
    
    #         else:
    #             pass
    #     except:
    #         pass


    # Drop Hits & Loand Date End because information not complete and not relevant
    df_fifa.drop(["Hits","Loan Date End"],axis=1,inplace=True)

    ## Delete Outliers 
    df_fifa.drop(outlier(df_fifa),axis=0,inplace=True)

    # Make a copy of the dataset
    df_fifa_work =  df_fifa.copy()

    # Encoding the categorical variable to numbers
    basic_encoding(df_fifa_work)

    # Convert the datetime to float
    df_fifa_work['Joined_float'] = dt64_to_float(df_fifa_work['Joined'].to_numpy())

    # Delete the categorical and Date Columns
    for i in df_fifa_work.columns:
            if df_fifa_work[i].dtype == 'object' or df_fifa_work[i].dtype == 'datetime64[ns]':
                    df_fifa_work.drop(i,axis=1,inplace=True)

    return df_fifa_work


def load_test_data (variable):
    df_test = pd.read_csv("./data/fifa21_test.csv", sep=",", low_memory=False, index_col=[0])
    df_fifa_predict = data_processing (df_test)
    df_fifa_predict.drop(columns=[variable],axis=1,inplace=True)
    # for col in df_fifa_predict.columns:
    #     print (f"{col} type",df_fifa_predict[col].dtype)

    return df_test,df_fifa_predict

def load_train_data ():
    df_train = pd.read_csv("./data/fifa21_train.csv", sep=",", low_memory=False, index_col=[0])
    df_fifa_train = data_processing (df_train)

    return df_train, df_fifa_train
    


## START MODEL
print ("---"*20)
print ("Football Player Price Estimation")
print ("---"*20)
opt = int(input('Enter 1 for run model OR Enter 2 for load model: \t '))
variable = input('Enter variable name: Wage | Value | Release Clause: \t ')

if opt == 1:
    # Load train data
    df_train, df_fifa_train = load_train_data ()

    print ('---'*20)
    print ("Modeling and Training is working")
    print ('---'*20)

    # Select the Data
    X = df_fifa_train.drop(columns=[variable], axis=1)
    y = df_fifa_train [[variable]]
   # X = X[:6000]
   # y = y[:6000]

    # Split Train, Test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=40)

    # Escalar Data
    escalar = MinMaxScaler()
    X_train_scale = escalar.fit_transform(X_train)
    X_test_scale = escalar.fit_transform(X_test)

    # Model TRAINING
    # Keras TF
    from gc import callbacks


    model_nn = keras.Sequential([
        layers.Dense(800, activation='relu', input_shape=X_train_scale.shape[1:]),
        layers.Dense(600, activation='selu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(200, activation='selu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='selu'),
        layers.Dense(1, activation='selu')
    ])

    model_nn.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                 metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanSquaredError()])
                #metrics=['mae','mse'])

    history_nn = model_nn.fit(X_train_scale, y_train, epochs=1000, validation_split=0.23, callbacks=keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10))
    df_hist = pd.DataFrame(history_nn.history)
    df_hist['epoch'] = history_nn.epoch

    model_nn.evaluate(X_test_scale, y_test)
    tiempo = datetime.today().strftime('%Y%m%d%H%M%S')
    model_nn.save(f"model_Fifa_NN__{tiempo}.h5")
    prediction = model_nn.predict(X_test_scale)



    
    print ('Modelling and Prediction successful')

    # Metrics
    print ('---'*20)
    print ("Metrics with known (training) data")
    print ('---'*20)
    print('RMSE on test data: ',  mean_squared_error(y_test, prediction)**0.5)
    print('MAPE%: ',  mean_absolute_percentage_error(y_test, prediction))
    print('MAE: ',  mean_absolute_error(y_test, prediction))
    print ('---'*20)

    ## Prediction STEPS

    # Data import for the Prediction Data and processing
    print ("---"*20)
    print ("Prediction for unseen data")
    df_test, df_fifa_predict =load_test_data(variable)

    # Scale and Do Prediction
    X_scal = escalar.fit_transform(df_fifa_predict) 
    y_pred_fin =  model_nn.predict(X_scal)



    tiempo = datetime.today().strftime('%Y%m%d%H%M%S')


    # Create Files
    print ('---'*20)
    print ('File creation')
    print ('---'*20)
    df_predictions = pd.DataFrame(y_pred_fin, columns=[variable])
    df_test[variable] = y_pred_fin
    df_fin = df_test
    df_fin.to_csv(f"./model/fifa21_prediction{tiempo}.csv",index=False)
    # PICKLE FILES
    filename = f'./model/finalized_model_{tiempo}.pkl'
    pickle.dump(model_nn, open(filename, 'wb'))
    data = df_test
    data.to_pickle(f'./model/datos_model_{tiempo}.dat')
    print ('---'*20)
    print ("Everything worked fine. Please check the files")

if opt == 2:
    print ('---'*20)
    print ('Load Model')
    print ('---'*20)
    filename_load = str(input('Enter the Pickle file name of the model: \t'))
    # load the model from disk
    if filename_load != "":
        try:
            pickled_model = pickle.load(open(filename_load, 'rb'))
            df_fifa_predict =load_test_data()
            X_scal = escalar.fit_transform(df_fifa_predict) 
            result = pickled_model.predict(X_scal)
            print(result)
        except:
            print ("Please check that the filename exists in the same folder")

