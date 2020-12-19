import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv

def cleanData(df):
    # df = pd.read_excel(filename)
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
    # df['PASSPORT'] = df['PASSPORT'] != "false"

    # marital status
    df['MARITAL'] = df['MARITAL'].fillna('NA')

    # death date
    df['DEATHDATE'] = df['DEATHDATE'].fillna('None-None-None')
    df[['DEATH_YEAR', 'DEATH_MONTH', 'DEATH_DAY']] = df['DEATHDATE'].str.split('-', 2, expand=True)
    df = df.drop(['DEATH_DAY'], axis =1)
    df = df.drop(['DEATHDATE'], axis =1)

    # extract city
    df['BIRTHPLACE'] = df['BIRTHPLACE'].str.split(' ', 0, expand=True)
    # print(df['BIRTHDATE'].str.split('-', 2, expand=True))

    return df


def getData():
    df_positive = pd.read_excel('For_30.xlsx')
    df_negative = pd.read_excel('Neg_For_30.xlsx')

    df_positive = df_positive.assign(target=1)
    df_negative = df_negative.assign(target=0)
    df = df_positive.append(df_negative, ignore_index=True)
    df = cleanData(df)
    return df
df = getData()
print(df.head(10))

# input()

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
# print(df_equalize_x.shape)
# print(df_equalize_y.shape)

sm = SMOTE(random_state =2)
x, y = sm.fit_sample(df_equalize_x, df_equalize_y.ravel())

x['GENDER'] = y
df = shuffle(x)

df = df.groupby(['GENDER'])
##################################################################################




from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(TP, TN, FP, FN):
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    # PPV = TP/(TP+FP)
    # NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    # FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return [TPR, TNR, FPR, FNR, ACC]


def compute_svm(data):
    global x
    global y
    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    # print(confusion_matrix(y_test,y_pred))
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # print(fpr)
    # print(tpr)
    # print(metrics.auc(fpr, tpr))
    # print("-----------------------x--------------------x---------")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return compute_metrics(tp, tn, fp, fn)

################ Linear Model ################

from sklearn import linear_model

def linearRegression(data):
    global x
    global y
    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    reg = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=0)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred<0,0,1)
    # print(confusion_matrix(y_test,y_pred))
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # print(tpr)
    # print(fpr)
    # print(metrics.auc(fpr, tpr))
    # print("-----------------------x--------------------x---------")
    # return [tpr[1], fpr[1], metrics.auc(fpr, tpr)]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return compute_metrics(tp, tn, fp, fn)


def logisticRegression(data):
    global x
    global y
    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)
    LR = linear_model.LogisticRegression(max_iter=500)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # print(tpr)
    # print(fpr)
    # print(metrics.auc(fpr, tpr))
    # print("-----------------------x--------------------x------------------")
    # return [tpr[1], fpr[1], metrics.auc(fpr, tpr)]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return compute_metrics(tp, tn, fp, fn)

for key,group in df:
    a = []
    b = []
    c = []
    for i in range(100):
        a.append(linearRegression(shuffle(group)))
        b.append(logisticRegression(shuffle(group)))
        c.append(compute_svm(shuffle(group)))
    print(np.average(a, axis=0))
    print(np.average(b, axis=0))
    print(np.average(c, axis=0))
    print("---------------x-------------------x------------")
    