import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import DecisionTreeClassifier
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
import numpy as np
import matplotlib.pyplot as plt
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
model1 = LogisticRegression(solver='liblinear',
                           C=0.05, multi_class='ovr'
                           ,random_state=0)
start_time = time.time()
model1.fit(x_train1, y_train1)
elapsed_time_training_1 = time.time() - start_time
print(f'{elapsed_time_training_1:.2f}s elapsed during training')
x_test1 = scaler.transform(x_test1)
start_time = time.time()
y_pred1 = model1.predict(x_test1)
elapsed_time_testing_1 = time.time() - start_time
print(f'{elapsed_time_testing_1:.2f}s elapsed during testing')
print("Training Model")
print("regression score",model1.score(x_train1, y_train1))
print("Testing Model")
print("regression score",model1.score(x_test1, y_test1))
print("Report Model 1 LogisticRegression")
print(classification_report(y_test1, y_pred1))
accuracy1=accuracy_score(y_test1, y_pred1)
pickle.dump(model1, open('model1_LogisticRegression.pkl', 'wb'))

###########################"Model 2"###############################

print("\n  Model 2  \n")

x_train2, x_test2, y_train2, y_test2 =train_test_split(X, Y, test_size=0.3,
                                                       random_state=1,
                                                       shuffle=True)
svm = svm.SVC(C=100.0, kernel='poly', degree=2)
start_time = time.time()
svm.fit(x_train2,y_train2)
elapsed_time_training_2 = time.time() - start_time
print(f'{elapsed_time_training_2:.2f}s elapsed during training')
print("training Model")
print("regression score",svm.score(x_train2,y_train2))
print("testing Model")
print("regression score",svm.score(x_test2,y_test2))
start_time = time.time()
y_pred2 = svm.predict(x_test2)
elapsed_time_testing_2 = time.time() - start_time
print(f'{elapsed_time_testing_2:.2f}s elapsed during testing')
print("Report Model 2 SVM")
print(classification_report(y_test2, y_pred2))
accuarcy2=accuracy_score(y_test2, y_pred2)
pickle.dump(svm, open('model2_SVM', 'wb'))

###########################"Model 3"###############################
print("\n  Model 3  \n")

x_train3, x_test3, y_train3, y_test3 =train_test_split(X, Y, test_size=0.3,
                                                       random_state=2,
                                                       shuffle=True)
clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                       algorithm="SAMME",n_estimators=200)
start_time = time.time()
clf.fit(x_train3, y_train3)
elapsed_time_training_3 = time.time() - start_time
print(f'{elapsed_time_training_3:.2f}s elapsed during training')
print("training Model")
AdaBoostClassifier(n_estimators=100, random_state=0)
print("regression score",clf.score(x_train3, y_train3))
print("testing Model")
print("regression score",clf.score(x_test3, y_test3))
start_time = time.time()
y_pred3 = clf.predict(x_test3)
elapsed_time_testing_3 = time.time() - start_time
print(f'{elapsed_time_testing_3:.2f}s elapsed during testing')
print("Report Model 3 DecisionTree")
print(classification_report(y_test3, y_pred3))
accuracy3=accuracy_score(y_test3, y_pred3)
pickle.dump(clf, open('model3_DecisionTreeClassifier.pkl', 'wb'))
##########################################

############################plot accuarcy###############################
# Dataset generation
data_dict = {'LogisticRegression':accuracy1, 'SVM':accuarcy2,'DecisionTreeClassifier':accuracy3}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig1 = plt.figure(figsize = (8, 8))
#  Bar plot
plt.bar(courses, values, color ='green',
        width = 0.1)
plt.xlabel("Models")
plt.ylabel("Acuuarcy")
plt.title(" Models & Accuarcy")
plt.show()
###############################plot training time########################
data_dict = {'LogisticRegression':elapsed_time_training_1,'SVM':elapsed_time_training_2 ,'DecisionTreeClassifier':elapsed_time_training_3}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig2 = plt.figure(figsize = (8, 8))
#  Bar plot
plt.bar(courses, values, color ='green',
        width = 0.1)
plt.xlabel("Models")
plt.ylabel("training time")
plt.title(" Models & Time_Training")
plt.show()
###############################plot testing time########################
data_dict = {'LogisticRegression':elapsed_time_testing_1, 'SVM':elapsed_time_testing_2,'DecisionTreeClassifier':elapsed_time_testing_3}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig3 = plt.figure(figsize = (8, 8))
#  Bar plot
plt.bar(courses, values, color ='green',
        width = 0.1)
plt.xlabel("Models")
plt.ylabel("testing time")
plt.title(" Models & Time_testing")
plt.show()
####################################################end plot###########################################################