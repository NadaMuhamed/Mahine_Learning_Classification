import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import *
import re
import time
from function import *
from statistics import mean
from fractions import Fraction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle
##########################################################
data = pd.read_csv('airline-price-classification.csv')
#data.dropna(how='any', inplace=True)
data['TicketCategory'] = data['TicketCategory'].replace(['cheap'],0)
data['TicketCategory'] = data['TicketCategory'].replace(['moderate'],1)
data['TicketCategory'] = data['TicketCategory'].replace(['expensive'],2)
data['TicketCategory'] = data['TicketCategory'].replace(['very expensive'],3)
##########################################################
X = data.iloc[:, 0:10]
Y = data.iloc[:, -1]
###################################################################
X = dictionary_to_columns(X, 'route')
cols = ('airline', 'ch_code', 'type', 'source', 'destination')
X = Feature_Encoder(X, cols)
X = Date_Converter(X)
X['time_taken'] = time_taken_to_seconds(X)
X['stop'] = Stop_Feature(X['stop'])
X['dep_time'] = converttomin(X['dep_time'])
X['arr_time'] = converttomin(X['arr_time'])
X['date'].fillna(mean(X['date']), inplace = True)
X['airline'].fillna(mean(X['airline']), inplace = True)
X['ch_code'].fillna(mean(X['airline']), inplace = True)
X['num_code'].fillna(mean(X['num_code']), inplace = True)
X['dep_time'].fillna(mean(X['dep_time']), inplace = True)
X['time_taken'].fillna(mean(X['time_taken']), inplace = True)
X['stop'].fillna(mean(X['airline']), inplace = True)
X['arr_time'].fillna(mean(X['arr_time']), inplace = True)
X['type'].fillna(mean(X['type']), inplace = True)
X['source'].fillna(mean(X['source']), inplace = True)
X['destination'].fillna(mean(X['destination']), inplace = True)
Y.fillna(mean(Y), inplace = True)
###########################"Model 1"###############################
print("\n  Model 1  \n")

x_train1, x_test1, y_train1, y_test1 =train_test_split(X, Y, test_size=0.3,
                                                       random_state=0,
                                                       shuffle=True)
scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
"""
model1=LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
                          intercept_scaling=1, l1_ratio=None, max_iter=100,
                          multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
                          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

model1=LogisticRegression(multi_class='multinomial', solver='lbfgs')

model1 = LogisticRegression(solver='liblinear',
                           C=0.05, multi_class='ovr'
                           ,random_state=0)
"""
model1=LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
                         intercept_scaling=1, l1_ratio=None, max_iter=100,
                         multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
                         solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
model1.fit(x_train1, y_train1)
x_test1 = scaler.transform(x_test1)
y_pred1 = model1.predict(x_test1)
print("Training Model")
print("regression score",model1.score(x_train1, y_train1))
print("Testing Model")
print("regression score",model1.score(x_test1, y_test1))
print("Report Model 1 LogisticRegression")
print(classification_report(y_test1, y_pred1))
pickle.dump(model1, open('model1_LogisticRegression.pkl', 'wb'))
###########################"Model 2"###############################
print("\n  Model 2  \n")

x_train2, x_test2, y_train2, y_test2 =train_test_split(X, Y, test_size=0.3,
                                                             random_state=1,
                                                             shuffle=True)
svm = svm.SVC(C=100.0, kernel='poly', degree=2)
svm.fit(x_train2,y_train2)
print("training Model")
print("regression score",svm.score(x_train2,y_train2))
print("testing Model")
print("regression score",svm.score(x_test2,y_test2))
print("Report Model 2 SVM")
y_pred2 = svm.predict(x_test2)
print(classification_report(y_test2, y_pred2))
pickle.dump(svm, open('model2_SVM', 'wb'))

###########################"Model 3"###############################
print("\n  Model 3  \n")

x_train3, x_test3, y_train3, y_test3 =train_test_split(X, Y, test_size=0.3,
                                                             random_state=88,
                                                             shuffle=True)
clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                       algorithm="SAMME",n_estimators=200)
AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(x_train3, y_train3)
print("training Model")
print("regression score",clf.score(x_train3, y_train3))
print("testing Model")
print("regression score",clf.score(x_test3, y_test3))
print("Report Model 3 DecisionTree")
y_pred3 = clf.predict(x_test3)
print(classification_report(y_test3, y_pred3))
pickle.dump(clf, open('model3_DecisionTreeClassifier.pkl', 'wb'))

####################################################################################################################
