# Data preprocessing tools

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


def dataPreprocess(dataset, ratio):
    categorical = {'breast-cancer.csv': True,
                   'diabetes.csv': False,
                   'heart.csv': False,
                   'image.csv': True,
                   'waveform.csv': False,
                   'glass_csv.csv': True,
                   'segment_csv.csv': True}

    # Importing the dataset
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Encoding the Dependent Variable
    if(categorical[dataset.split('/')[-1]]):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    return train_test_split(X, y, test_size=ratio, random_state=1)


def builder(dataset, ratio):
    X_train, X_test, y_train, y_test = dataPreprocess(dataset, ratio)
    classification_reports = dict()
    model_report = dict()
    models = ['Perceptron', 'SVC',
              'Gaussian NB', 'Decision Tree', 'Gradient Boosting']
    time_taken = []
    accuracy = []
    precision = []
    recall = []
    f1 = []

    # PERCEPTRON
    # -------------
    start_perceptron = time.time()
    model = Perceptron()
    model.fit(X_train, y_train)
    y_predP = model.predict(X_test)
    cr = classification_report(y_test, y_predP)

    l = cr.split("\n")
    acc = l[-4].split()

    accuracy.append(float(acc[1]))
    x = l[-2].split()
    precision.append(float(x[2]))
    recall.append(float(x[3]))
    f1.append(float(x[4]))
    total_time_perceptron = (time.time() - start_perceptron)
    time_taken.append(total_time_perceptron)
    model_report['Perceptron'] = [
        float(acc[1]), float(x[2]), float(x[3]), float(x[4])]

    # SUPPORT VECTOR CLASSIFIER
    # ----------------------------
    start_svc = time.time()
    model = SVC()
    model.fit(X_train, y_train)
    y_predS = model.predict(X_test)
    cr = classification_report(y_test, y_predS)

    l = cr.split("\n")
    acc = l[-4].split()
    accuracy.append(float(acc[1]))
    x = l[-2].split()
    precision.append(float(x[2]))
    recall.append(float(x[3]))
    f1.append(float(x[4]))
    total_time_svc = (time.time() - start_svc)
    time_taken.append(total_time_svc)
    model_report['SVC'] = [float(acc[1]), float(
        x[2]), float(x[3]), float(x[4])]

    # GAUSSIAN NAIVE BAYES
    # ------------------------
    start_gnb = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_predG = model.predict(X_test)
    cr = classification_report(y_test, y_predG)

    l = cr.split("\n")
    acc = l[-4].split()
    accuracy.append(float(acc[1]))
    x = l[-2].split()
    precision.append(float(x[2]))
    recall.append(float(x[3]))
    f1.append(float(x[4]))
    total_time_gnb = (time.time() - start_gnb)
    time_taken.append(total_time_gnb)
    model_report['GaussianNB'] = [
        float(acc[1]), float(x[2]), float(x[3]), float(x[4])]

    # DECISION TREE CLASSIFIER
    # ----------------------------
    start_dt = time.time()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predD = model.predict(X_test)
    cr = classification_report(y_test, y_predD)

    l = cr.split("\n")
    acc = l[-4].split()
    accuracy.append(float(acc[1]))
    x = l[-2].split()
    precision.append(float(x[2]))
    recall.append(float(x[3]))
    f1.append(float(x[4]))
    total_time_dt = (time.time() - start_dt)
    time_taken.append(total_time_dt)
    model_report['Decision Tree'] = [
        float(acc[1]), float(x[2]), float(x[3]), float(x[4])]

    # GRADIENT BOOSTING CLASSIFIER
    # --------------------------------
    start_xgb = time.time()
    graClassifier = GradientBoostingClassifier()
    graClassifier.fit(X_train, y_train)
    y_pred = graClassifier.predict(X_test)
    cr = classification_report(y_test, y_pred)
    #

    l = cr.split("\n")
    acc = l[-4].split()
    accuracy.append(float(acc[1]))
    x = l[-2].split()
    precision.append(float(x[2]))
    recall.append(float(x[3]))
    f1.append(float(x[4]))
    total_time_xgb = (time.time() - start_xgb)
    time_taken.append(total_time_xgb)
    model_report['Gradient Boosting'] = [
        float(acc[1]), float(x[2]), float(x[3]), float(x[4])]

    # ADDING ALL MODELS INTO THE REPORT
    classification_reports = {"models": models, "accuracy": accuracy, "precision": precision,
                              "recall": recall, "f1": f1, "training_time": time_taken}
    # print(classification_reports)
    return classification_reports, model_report

# builder('https://raw.githubusercontent.com/bullet-ant/algorithmComparisionApp/main/Datasets/diabetes.csv', 0.2)
