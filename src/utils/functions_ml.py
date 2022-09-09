import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def nullos (df):
    """ shows number of nans/nulls """
    for i in df.columns:
        print (f"nan in Column  {i} :",df[i].isna().sum())
        print (f"null in Column  {i} :",df[i].isnull().sum())


def field_cleanerx (df,col,replacestring):
    """ Replaces a string as kg with nonspace """
    df[col] = df[col].apply(lambda x : x.replace(replacestring,"") )



def convert_inch_in_row (lista_inches,df,field): 
    """ Converts inch in cm in each row """
    for i in lista_inches:
    #print(df.loc[i,field])
        changeable = df.loc[i,field].replace('"','').split("'")
        foot = 30.48
        inch = 2.54
        val1 = (float(changeable[0])*foot)
        val2 = (float(changeable[1])*inch)
    # print (round(val1+val2,2))
        df.loc[i,field] = round(val1+val2,2)


def convert_lbs_to_kg (lista_libs,df,field):
    """ Converts pounds in kg in each row """
    for i in lista_libs:
        changeable = df.loc[i,field].replace('lbs','')
        b = 0.45359237 
        val1 = (float(changeable)*b)
        df.loc[i,field] = round(val1,2)



def convert_monetary (x):
    """ Replaces EURO Symbol with nonspace in order to use numbers """
    x = x.replace("€","")

    if "M" in x:
        x = float(x.replace("M",""))*1000000
    elif "K" in x:
        x = float(x.replace("K",""))*1000
    else:
        pass

    return  x

def force_col_convert (listnames,df,type):
    """ Forces conversion of columns to a type for example float, str etc."""
    for i in listnames:
        df[i] = df[i].astype(type)


def outlier (df):
    """ returns a list with indices of all outliers"""
    for col in df:
        if df[col].dtype == "float64" or df[col].dtype == 'int64':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_outliers =  df[df[col] < (Q1 - 1.5 * IQR)].index.tolist()
            upper_outliers =  df[df[col] > (Q3 + 1.5 * IQR)].index.tolist()
            bad_indices = list(set(upper_outliers + lower_outliers))
            return bad_indices



def basic_encoding (df):
    """Does a basic lable encoding of categorical variables"""
    le = LabelEncoder()
    for i in df.columns:
            if df[i].dtype == 'object':
                    enc_name = i+"_encoded"
                    df[enc_name] = le.fit_transform(df[i])

    return df


def dt64_to_float(dt64):
   """ Converts a datetime64 format to float"""
   year = dt64.astype('M8[Y]')
   days = (dt64 - year).astype('timedelta64[D]')
   year_next = year + np.timedelta64(1, 'Y')
   days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')).astype('timedelta64[D]')
   dt_float = 1970 + year.astype(float) + days / (days_of_year)
   return dt_float

