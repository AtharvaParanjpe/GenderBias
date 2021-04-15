import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv
import matplotlib.pyplot as plt  
import seaborn as sns

flag_svm = False
flag_decision_tree = False
flag_linear = False
flag_logistic = False

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
    df_positive = pd.read_excel('For_30.xlsx')
    df_positive["Age"] = df_positive["SUICIDE_AGE"] 
    df_positive = df_positive.drop(["SUICIDE_AGE"],axis = 1)

    df_negative = pd.read_excel('Neg_For_30.xlsx')
    df_negative["Age"] = df_negative["SUICIDE_AGE"] 
    df_negative = df_negative.drop(["SUICIDE_AGE"],axis = 1)
    

    df_positive = df_positive.assign(target=1)
    df_negative = df_negative.assign(target=0)
    df = df_positive.append(df_negative, ignore_index=True)
    
    # plot the data 
    p = sns.countplot(data=df, x = 'target', hue="GENDER")
    # p.legend([],[], frameon=False)
    p.plot()
    # p.show()
    
    df = shuffle(df)
    df = cleanData(df)
    return df

# def scatterPlot(x_data,y_data):
#     plt.figure()
#     for x, y in zip(x_data, y_data):
#         color = "green" if y == 0 else "red"
#         plt.scatter(x, y, color = color)
#         plt.xlabel('Data points')
#         plt.ylabel('Suicidality')
#     plt.show()


df = getData()
# print(df.head(5))


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

df = df.groupby(['GENDER'])

##################################################################################


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(TP, TN, FP, FN, auc):
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    # PPV = TP/(TP+FP)
    # NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    # FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # print(TPR, TNR, FPR, FNR, ACC, auc)
    return [TPR, TNR, FPR, FNR, ACC, auc]


def compute_svm(data):
    global flag_svm
    
    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    # print(confusion_matrix(y_test,y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    if(flag_svm):
        flag_svm = False
        metrics.plot_roc_curve(svclassifier, X_test, y_test) 
        plt.show()
    auc = metrics.auc(fpr, tpr)
    return compute_metrics(tp, tn, fp, fn, auc)

################ Linear Model ################

from sklearn import linear_model, tree

def linearRegression(data):
    global flag_linear

    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    reg = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred<0,0,1)
    # print(confusion_matrix(y_test,y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    return compute_metrics(tp, tn, fp, fn, auc)


def logisticRegression(data):
    global flag_logistic

    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    LR = linear_model.LogisticRegression(max_iter=1000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    auc = metrics.auc(fpr, tpr)
    if(flag_logistic):
        flag_logistic = False
        metrics.plot_roc_curve(LR, X_test, y_test) 
        plt.show()
    return compute_metrics(tp, tn, fp, fn, auc)

def decision_tree(data):
    global flag_decision_tree

    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    if(flag_decision_tree):
        flag_decision_tree = False
        metrics.plot_roc_curve(clf, X_test, y_test) 
        plt.show()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    auc = metrics.auc(fpr, tpr)
    return compute_metrics(tp, tn, fp, fn, auc)
    

# print("TPR, TNR, FPR, FNR, Accuracy, AUC")
final_list = []

for key,group in df:
    group = group.drop(["GENDER"], axis=1)
    a = []
    b = []
    c = []
    d = []

    # flag_svm = True
    # flag_decision_tree = True
    # flag_linear = True
    flag_logistic = True

    group = shuffle(group)
    for i in range(100):
        a.append(linearRegression(group))
        b.append(logisticRegression(group))
        c.append(compute_svm(group))
        d.append(decision_tree(group))
        # input()
    
    a = np.average(a, axis=0).tolist()
    b = np.average(b, axis=0).tolist()
    c = np.average(c, axis=0).tolist()
    d = np.average(d, axis=0).tolist()

    a = ["Linear_"+str(key)] + a
    b = ["Logistic_"+str(key)] + b
    c = ["SVM_"+str(key)] + c
    d = ["Decision Tree_"+str(key)] + d

    final_list = final_list+[a]+[b]+[c]+[d]
    

columns = ["Model", "TPR", "TNR", "FPR", "FNR", "Accuracy", "AUC"]
df = pd.DataFrame(data= final_list, columns=columns)
print(df)

