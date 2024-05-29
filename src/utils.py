import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def del_outlier(df):
    return df

def encode(df):
    df = pd.get_dummies(df)
    return df


def split(df, method_split, target_colname=False):
    if method_split=='8_2' or method_split=='CV_10':
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    if method_split=='Doctor':
        if os.path.exists('./data/ID_doctorjudge.csv')==False:
            print('No file "./data/ID_doctorjudge.csv"')
            return False
        df_doctorjudge = pd.read_csv('./data/ID_doctorjudge.csv')
        list_ID_doctorjudge = df_doctorjudge.ID.tolist()
        mask = df['ID'].isin(list_ID_doctorjudge)
        df_test = df[mask]
        df_train = df[~mask]
    if method_split=='StartifiedS': # Startified Sampling
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_colname])
    return df_train, df_test

def norm(X_train, X_test, method_norm):
    if method_norm==False:
        pass
    if method_norm=='SC':
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    return X_train, X_test


def do_preprocess(df, is_del_outlier, method_norm, method_split, target_colname, size_name, length_name, is_cla=False):
    
    if is_del_outlier:
        del_outlier(df)
    
    df = encode(df)

    df_train, df_test = split(df, method_split, target_colname)
    print(df_train)
    print(df_test)
    X_train = df_train.drop(['ID', size_name, length_name], axis=1)    
    X_test = df_test.drop(['ID', size_name, length_name], axis=1)    
    X_train, X_test = norm(X_train, X_test, method_norm)

    y_train = df_train.loc[:, target_colname].values
    y_test = df_test.loc[:, target_colname].values

    return df_train, df_test, X_train, X_test, y_train, y_test