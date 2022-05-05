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
data = pd.read_csv('airline-price-prediction.csv')
data.dropna(how='any', inplace=True)
#data['date'] = data['date'].replace([''],'13-03-2022')
##########################################################
X = data.iloc[:, 0:10]
Y = handel_price(data['price'])
X = dictionary_to_columns(X, 'route')
cols = ('airline', 'ch_code', 'type', 'source', 'destination','TicketCategory')
X = Feature_Encoder(X, cols)
X = Date_Converter(X)
X['time_taken'] = time_taken_to_seconds(X)
X['stop'] = Stop_Feature(X['stop'])
X['dep_time'] = converttomin(X['dep_time'])
X['arr_time'] = converttomin(X['arr_time'])
airline = X
airline['price'] = Y

###########################"Model 1"###############################
print("\n  Model 1  \n")

###########################"Model 2"###############################
print("\n  Model 2  \n")

###########################"Model 3"###############################
print("\n  Model 3  \n")

###########################"Model 4"###############################
print("\n  Model 4  \n")

####################################################################################################################
