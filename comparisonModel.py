## For Generating comparison graphs between datasets

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv
import matplotlib.pyplot as plt  
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

flag_svm = False
flag_decision_tree = False
flag_linear = False
flag_logistic = False

## Generate a file with min age 10 to proceed
df = pd.read_excel('Final_File_min10.xlsx')
df2 = pd.read_excel('Final_File_min13.xlsx')
# print(df.head())
# input()
df = df.groupby(['GENDER'])
df2 = df2.groupby(['GENDER'])

def scatterPlot(x_data,y_data):
    plt.figure()
    for x, y in zip(x_data, y_data):
        color = "green" if y == 0 else "red"
        plt.scatter(x, y, color = color)
        plt.xlabel('Data points')
        plt.ylabel('Suicidality')
    plt.show()

def compute_metrics(TP, TN, FP, FN, auc):
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return [TPR, TNR, FPR, FNR, ACC, auc]

def compute_svm(data, data2):
    global flag_svm
    
    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    if(flag_svm):
        # flag_svm = False
        plt.figure(0).clf()
        plt.title("Support Vector Classifier")
        # metrics.plot_roc_curve(svclassifier, X_test, y_test) 
        plt.plot(fpr,tpr,label="For min age 10, Auc="+str(round(auc, 4)))
        # plt.show()
    
    
    y = data2['target']
    x = data2
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    if(flag_svm):
        flag_svm = False
        # metrics.plot_roc_curve(svclassifier, X_test, y_test) 
        plt.plot(fpr,tpr,label="For min age 13, Auc="+str(round(auc,4)))
        plt.legend(loc=0)
        plt.show()
    
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
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    return compute_metrics(tp, tn, fp, fn, auc)

def logisticRegression(data, data2):
    global flag_logistic

    plt.figure(0).clf()
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
        # flag_logistic = False
        plt.title("Logistic Classifier")
        plt.plot(fpr,tpr,label="For min age 10, Auc="+str(round(auc,4)))
        # plt.show()

    y = data2['target']
    x = data2
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    LR = linear_model.LogisticRegression(max_iter=1000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    if(flag_logistic):
        flag_logistic = False
        # metrics.plot_roc_curve(svclassifier, X_test, y_test) 
        plt.plot(fpr,tpr,label="For min age 13, Auc="+str(round(auc,4)))
        plt.legend(loc=0)
        plt.show()

    return compute_metrics(tp, tn, fp, fn, auc)

def decision_tree(data, data2):
    global flag_decision_tree

    y = data['target']
    x = data
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    if(flag_decision_tree):
        plt.figure(0).clf()
        plt.title("Decision Tree Classifier")
        # flag_decision_tree = False
        # metrics.plot_roc_curve(clf, X_test, y_test) 
        plt.plot(fpr,tpr,label="For min age 10, Auc="+str(round(auc,4)))
        # plt.show()

    y = data2['target']
    x = data2
    x = x.drop('target', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    if(flag_decision_tree):
        flag_decision_tree = False
        # metrics.plot_roc_curve(clf, X_test, y_test) 
        plt.plot(fpr,tpr,label="For min age 13, Auc="+str(round(auc,4)))
        # plt.show()
        plt.legend(loc=0)
        plt.show()
    return compute_metrics(tp, tn, fp, fn, auc)
    
final_list = []

for (key,group),(key1,group1) in zip(df, df2):
    group = group.drop(["GENDER"], axis=1)
    group1 = group1.drop(["GENDER"], axis=1)
    print(key1)

    a = []
    b = []
    c = []
    d = []

    flag_svm = True
    flag_decision_tree = True
    flag_linear = True
    flag_logistic = True

    group = shuffle(group)
    group1 = shuffle(group1)

    for i in range(100):
        a.append(linearRegression(group))
        b.append(logisticRegression(group, group1))
        c.append(compute_svm(group, group1))
        d.append(decision_tree(group, group1))
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

## Uncomment to generate final results
# df.to_excel('Result_with_min_13.xlsx', index=False, header=True)

