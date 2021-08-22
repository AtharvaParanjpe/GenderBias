import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv
import matplotlib.pyplot as plt  
import seaborn as sns

def cleanData(df):
    df = df.drop(['SSN', 'DRIVERS', 'PREFIX', 'FIRST', 'LAST', 'SUFFIX', 'MAIDEN', 'ADDRESS'], axis=1)
    patientDictionary = {}
    for key,value in enumerate(df['ID'].values):
        patientDictionary[value] = key
    
    #ID
    df['ID'] = df['ID'].replace(patientDictionary)

    # DOB
    df[['BIRTH_YEAR', 'BIRTH_MONTH', 'BIRTH_DAY']] = df['BIRTHDATE'].str.split('-', 2, expand=True)
    df = df.drop(['BIRTH_DAY'], axis = 1)
    df = df.drop(['BIRTHDATE'], axis = 1)

    # Yes no encoding
    values = []
    for x in df['PASSPORT'].values:
        if(x!="false"):
            values.append("Y")
        else:
            values.append("N")
    df['PASSPORT'] = values
    
    # marital status
    df['MARITAL'] = df['MARITAL'].fillna('NA')

    # death date
    df['DEATHDATE'] = df['DEATHDATE'].fillna('None-None-None')
    df[['DEATH_YEAR', 'DEATH_MONTH', 'DEATH_DAY']] = df['DEATHDATE'].str.split('-', 2, expand=True)
    df = df.drop(['DEATH_DAY'], axis =1)
    df = df.drop(['DEATHDATE'], axis =1)

    # extract city
    df['BIRTHPLACE'] = df['BIRTHPLACE'].str.split(' ', 0, expand=True)
    
    return df


def getData():
    df_positive = pd.read_excel('./pos/For_30.xlsx')
    df_positive["Age"] = df_positive["SUICIDE_AGE"] 
    df_positive = df_positive.drop(["SUICIDE_AGE"],axis = 1)

    df_negative = pd.read_excel('./neg/Neg_For_30_min13.xlsx')
    df_negative["Age"] = df_negative["SUICIDE_AGE"] 
    df_negative = df_negative.drop(["SUICIDE_AGE"],axis = 1)
    

    df_positive = df_positive.assign(target=1)
    df_negative = df_negative.assign(target=0)
    df = df_positive.append(df_negative, ignore_index=True)
    
    df = shuffle(df)
    df = cleanData(df)
    return df

df = getData()
df.to_excel('Final_File_before_label_encoding.xlsx', index=False, header=True)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def preProcessData(df):
    # Categorical boolean mask
    categorical_feature_mask = df.dtypes==object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

df_labelEncoded = preProcessData(df)


############################ SMOTE TO BALANCE THE DATA ###########################

from imblearn.over_sampling import SMOTE

df_equalize_y = df_labelEncoded['GENDER']
df_equalize_x = df_labelEncoded.drop('GENDER', axis = 1)

sm = SMOTE(random_state =2)
x, y = sm.fit_sample(df_equalize_x, df_equalize_y.ravel())

x['GENDER'] = y
df = shuffle(x)

df.to_excel('Final_File_min13.xlsx', index=False, header=True)

