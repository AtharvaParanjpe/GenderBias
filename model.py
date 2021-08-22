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
    distribution_list = []
    a, b, c, d = [], [], [], []

    flag_svm = True
    flag_decision_tree = True
    flag_linear = True
    flag_logistic = True

    group = shuffle(group)

    for i in range(1000):
        final_list = []

        linear = linearRegression(group)
        logistic = logisticRegression(group)
        svc = compute_svm(group)
        dt = decision_tree(group)

        a.append([linear[0]])
        b.append([logistic[0]])
        c.append([svc[0]])
        d.append([dt[0]])

        distribution_list.append([linear[0][-2], logistic[0][-2], svc[0][-2], dt[0][-2]])
        # print([linear[0][-2], logistic[0][-2], svc[0][-2], dt[0][-2]]) 
        # input()

    a_mean = np.average(a, axis=0).tolist()
    b_mean = np.average(b, axis=0).tolist()
    c_mean = np.average(c, axis=0).tolist()
    d_mean = np.average(d, axis=0).tolist()
   
    a_min = np.min(a, axis=0).tolist()
    b_min = np.min(b, axis=0).tolist()
    c_min = np.min(c, axis=0).tolist()
    d_min = np.min(d, axis=0).tolist()

    a_max = np.max(a, axis=0).tolist()
    b_max = np.max(b, axis=0).tolist()
    c_max = np.max(c, axis=0).tolist()
    d_max = np.max(d, axis=0).tolist()

    a_mean = ["Linear"] + a_mean[0]
    b_mean = ["Logistic"] + b_mean[0]
    c_mean = ["SVM"] + c_mean[0]
    d_mean = ["Decision Tree"] + d_mean[0]

    a_min = ["Linear"] + a_min[0]
    b_min = ["Logistic"] + b_min[0]
    c_min = ["SVM"] + c_min[0]
    d_min = ["Decision Tree"] + d_min[0]

    a_max = ["Linear"] + a_max[0]
    b_max = ["Logistic"] + b_max[0]
    c_max = ["SVM"] + c_max[0]
    d_max = ["Decision Tree"] + d_max[0]

    column_names_for_distribution = ["Linear", "Logistic", "SVM", "Decision Tree"]
    columns = ["Model", "TPR", "TNR", "FPR", "FNR", "Accuracy", "AUC"]
    
    distribution = pd.DataFrame(data= distribution_list, columns=column_names_for_distribution)
    distribution.to_excel('./Temporary Distribution/Data_Points_For_Accuracy_' + str(key) + '.xlsx', index=False, header=True)

    final_list = [a_mean, b_mean, c_mean, d_mean]
    
    mean_distribution = pd.DataFrame(data= final_list, columns=columns)
    mean_distribution.to_excel('./Temporary Distribution/Mean Distibution - ' + str(key) + '.xlsx', index=False, header=True)

    final_list = [a_min, b_min, c_min, d_min]
    min_distribution = pd.DataFrame(data= final_list, columns=columns)
    min_distribution.to_excel('./Temporary Distribution/Min Distibution - ' + str(key) + '.xlsx', index=False, header=True)

    final_list = [a_max, b_max, c_max, d_max]
    max_distribution = pd.DataFrame(data= final_list, columns=columns)
    max_distribution.to_excel('./Temporary Distribution/Max Distibution - ' + str(key) + '.xlsx', index=False, header=True)

 