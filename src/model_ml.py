
# MAIN LIBS
import os
import pandas as pd
import numpy as np

import sys


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



## START MODEL


# Load train data
df_train = pd.read_csv("./data/fifa21_train.csv", sep=",", low_memory=False, index_col=[0])
df_fifa_train = data_processing (df_train)

# conversion type check
for col in df_fifa_train.columns:
    print (f"{col} type",df_fifa_train[col].dtype)

# Select the Data
X = df_fifa_train.drop(columns=['Value'], axis=1)
y = df_fifa_train [['Value']]
X = X[:1600]
y = y[:1600]


# Split Train, Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=40)

# Escalar Data
escalar = MinMaxScaler()
X_train_scale = escalar.fit_transform(X_train)
X_test_scale = escalar.fit_transform(X_test)

# Model 
rf = RandomForestRegressor( n_estimators = 400,
                criterion    = 'absolute_error',
                random_state = 40)

rf.fit(X_train_scale, y_train.values.ravel())
prediction  = rf.predict(X_test_scale)

print(prediction)
# Metrics
print ('---'*20)
print ("Train")
print ('---'*20)
print('RMSE on test data: ',  mean_squared_error(y_test, prediction)**0.5)
print('MAPE%: ',  mean_absolute_percentage_error(y_test, prediction))
print('MAE: ',  mean_absolute_error(y_test, prediction))
print ('---'*20)

## Prediction STEPS

# Data import and processing
print ("---"*20)
print ("PREDICTION")
df_test = pd.read_csv("./data/fifa21_test.csv", sep=",", low_memory=False, index_col=[0])
df_fifa_predict = data_processing (df_test)
df_fifa_predict.drop(columns=["Value"],axis=1,inplace=True)
for col in df_fifa_predict.columns:
    print (f"{col} type",df_fifa_predict[col].dtype)

# Scale and apply Model
X_scal = escalar.fit_transform(df_fifa_predict) 
y_pred_fin =  rf.predict(X_scal)
print ('---'*20)
print ("Predictions")
print ('---'*20)
print(y_pred_fin)
print ('---'*20)
tiempo = datetime.today().strftime('%Y%m%d%H%M%S')
print(y_pred_fin.shape)

# Create File
df_predictions = pd.DataFrame(y_pred_fin, columns=["Value"])
df_test.drop(columns=["Value"],axis=1,inplace=True)

# forming dataframe final and printing
df_fin = pd.concat([df_test,df_predictions], axis=1)
df_fin.to_csv(f"./model/fifa21_prediction{tiempo}.csv")
data = df_predictions
print(data) 
# using to_pickle function to form file 
# with name 'pickle_file'
data.to_pickle(f'./model/modelo{tiempo}')