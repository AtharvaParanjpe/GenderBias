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


df = pd.read_excel('Final_File_min13.xlsx')
df = df.groupby(['GENDER'])

def scatterPlot(x_data,y_data):
    plt.figure()
    for x, y in zip(x_data, y_data):
        color = "green" if y == 0 else "red"
        plt.scatter(x, y, color = color)
        plt.xlabel('Data points')
        plt.ylabel('Suicidality')
    plt.show()

def compute_metrics(TP, TN, FP, FN, auc, fpr, tpr):
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return [TPR, TNR, FPR, FNR, ACC, auc], fpr, tpr

def compute_svm(data):
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
    if(len(fpr)==2):
        fpr = [fpr[0], 0.5, fpr[1]]
        tpr = [tpr[0], 0.5, tpr[1]]
    return compute_metrics(tp, tn, fp, fn, auc, fpr, tpr)

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
    if(len(fpr)==2):
        fpr = [fpr[0], 0.5, fpr[1]]
        tpr = [tpr[0], 0.5, tpr[1]]
    return compute_metrics(tp, tn, fp, fn, auc, fpr, tpr)

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
    if(len(fpr)==2):
        fpr = [fpr[0], 0.5, fpr[1]]
        tpr = [tpr[0], 0.5, tpr[1]]
    return compute_metrics(tp, tn, fp, fn, auc, fpr, tpr)

def decision_tree(data):
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
    if(len(fpr)==2):
        fpr = [fpr[0], 0.5, fpr[1]]
        tpr = [tpr[0], 0.5, tpr[1]]
    return compute_metrics(tp, tn, fp, fn, auc, fpr, tpr)
    
final_list = []

def showGraph(fpr, tpr, auc, classifier):
    plt.figure()
    plt.plot(fpr, tpr,label= classifier + ", Auc="+str(round(auc,4)))
    plt.legend(loc=0)
    plt.show()

for (key,group) in df:
    group = group.drop(["GENDER"], axis=1)
    
    a, a_fpr, a_tpr = [], [], []
    b, b_fpr, b_tpr = [], [], []
    c, c_fpr, c_tpr = [], [], []
    d, d_fpr, d_tpr = [], [], []

    flag_svm = True
    flag_decision_tree = True
    flag_linear = True
    flag_logistic = True

    group = shuffle(group)

    for i in range(100):

        linear = linearRegression(group)
        logistic = logisticRegression(group)
        svc = compute_svm(group)
        dt = decision_tree(group)

        a.append(linear[0])
        b.append(logistic[0])
        c.append(svc[0])
        d.append(dt[0])

        a_fpr.append(linear[1])
        a_tpr.append(linear[2])
        b_fpr.append(logistic[1])
        b_tpr.append(logistic[2])
        c_fpr.append(svc[1])
        c_tpr.append(svc[2])
        d_fpr.append(dt[1])
        d_tpr.append(dt[2])
    
    a = np.average(a, axis=0).tolist()
    b = np.average(b, axis=0).tolist()
    c = np.average(c, axis=0).tolist()
    d = np.average(d, axis=0).tolist()

    a_fpr = np.average(a_fpr, axis=0).tolist()
    a_tpr = np.average(a_tpr, axis=0).tolist()
    b_fpr = np.average(b_fpr, axis=0).tolist()
    b_tpr = np.average(b_tpr, axis=0).tolist()
    c_fpr = np.average(c_fpr, axis=0).tolist()
    c_tpr = np.average(c_tpr, axis=0).tolist()
    d_fpr = np.average(d_fpr, axis=0).tolist()
    d_tpr = np.average(d_tpr, axis=0).tolist()

    showGraph(d_fpr, d_tpr, d[-1], "Decision Tree")
    showGraph(b_fpr, b_tpr, b[-1], "Logistic Regression")
    showGraph(c_fpr, c_tpr, c[-1], "SVC")
    

    a = ["Linear_"+str(key)] + a
    b = ["Logistic_"+str(key)] + b
    c = ["SVM_"+str(key)] + c
    d = ["Decision Tree_"+str(key)] + d

    final_list = final_list+[a]+[b]+[c]+[d]

columns = ["Model", "TPR", "TNR", "FPR", "FNR", "Accuracy", "AUC"]
df = pd.DataFrame(data= final_list, columns=columns)

## Uncomment to generate final results
# c = input("Save this file?")
# if(c=="Y" or c=="y"):
#     df.to_excel('Result_with_min_13(4).xlsx', index=False, header=True)

