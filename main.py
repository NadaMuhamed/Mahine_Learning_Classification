import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import *
import re
import time
from function import *
from statistics import mean
from fractions import Fraction as fr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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
X['TicketCategory']=Y
airline = X
airline['TicketCategory'] = Y
###########################"Model 1"###############################
print("\n  Model 1  \n")
x_train1, x_test1, y_train1, y_test1 =train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)
scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train1)
model1 = LogisticRegression(solver='liblinear',
                           C=0.05, multi_class='ovr'
                           ,random_state=0)
model1.fit(x_train1, y_train1)
LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
x_test1 = scaler.transform(x_test1)
y_pred1 = model1.predict(x_test1)
print("Training Model")
print("regression score",model1.score(x_train1, y_train1))
print("Testing Model")
print("regression score",model1.score(x_test1, y_test1))
print('Mean Square Error', metrics.mean_squared_error(y_test1, y_pred1))

confusion_matrix(y_test1, y_pred1)
cm = confusion_matrix(y_test1, y_pred1)
print("Report Model 1 LogisticRegression")
print(classification_report(y_test1, y_pred1))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=20, color='black')
ax.set_ylabel('Actual outputs', fontsize=20, color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_ylim(9.5, -0.5)
"""
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
"""
ax.set_title("airline price classification")
fig.tight_layout()
#plt.show()

###########################"Model 2"###############################
print("\n  Model 2  \n")
x_train2, x_test2, y_train2, y_test2 =train_test_split(X, Y, test_size=0.3, random_state=0,shuffle=True)
y_pred2 = model1.predict(x_test2)
svm = svm.SVC(C=1.0, kernel='poly', degree=25)
svm.fit(x_train2,y_train2)
print("training Model")
print("regression score",svm.score(x_test2,y_test2))
print("testing Model")
print("regression score",svm.score(x_test2,y_test2))
print("Report Model 2 SVM")
print(classification_report(y_test2, y_pred2))
###########################"Model 3"###############################
print("\n  Model 3  \n")

###########################"Model 4"###############################
print("\n  Model 4  \n")

####################################################################################################################
