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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from statistics import mean
from fractions import Fraction as fr
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
airline = X
airline['TicketCategory'] = Y

###########################"Model 1"###############################
print("\n  Model 1  \n")
model1 = LogisticRegression(solver='liblinear', random_state=0)
###########################"Model 2"###############################
print("\n  Model 2  \n")

###########################"Model 3"###############################
print("\n  Model 3  \n")

###########################"Model 4"###############################
print("\n  Model 4  \n")

####################################################################################################################
