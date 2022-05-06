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
##########################################################
data = pd.read_csv('airline-price-classification.csv')
#data.dropna(how='any', inplace=True)
data['date'].fillna("30-03-2022", inplace = True)
data['airline'].fillna("Vistara", inplace = True)
data['ch_code'].fillna("UK", inplace = True)
data['num_code'].fillna(2485, inplace = True)
data['dep_time'].fillna("12:00", inplace = True)
data['time_taken'].fillna("13h 15m", inplace = True)
data['stop'].fillna("1-stop", inplace = True)
data['arr_time'].fillna("11:59:00 PM", inplace = True)
data['type'].fillna("economy", inplace = True)
data['route'].fillna("{'source': 'Delhi', 'destination': 'Kolkata'}", inplace = True)
data['TicketCategory'].fillna("moderate", inplace = True)
data['TicketCategory'] = data['TicketCategory'].replace(['cheap'],0)
data['TicketCategory'] = data['TicketCategory'].replace(['moderate'],1)
data['TicketCategory'] = data['TicketCategory'].replace(['expensive'],2)
data['TicketCategory'] = data['TicketCategory'].replace(['very expensive'],3)
##########################################################
X = data.iloc[:, 0:10]
Y = data.iloc[:, -1]
X = dictionary_to_columns(X, 'route')
cols = ('airline', 'ch_code', 'type', 'source', 'destination')
X = Feature_Encoder(X, cols)
X = Date_Converter(X)
X['time_taken'] = time_taken_to_seconds(X)
X['stop'] = Stop_Feature(X['stop'])
X['dep_time'] = converttomin(X['dep_time'])
X['arr_time'] = converttomin(X['arr_time'])
airline = X
airline['TicketCategory'] = Y
print(Y)
###########################"Model 1"###############################
print("\n  Model 1  \n")

###########################"Model 2"###############################
print("\n  Model 2  \n")

###########################"Model 3"###############################
print("\n  Model 3  \n")

###########################"Model 4"###############################
print("\n  Model 4  \n")

####################################################################################################################
