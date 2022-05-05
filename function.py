from sklearn import svm, datasets
import numpy as np
import pandas as pd
import seaborn as sns
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
import ast
import math
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder
import re



def time_taken_to_seconds(X):
    UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    return [int(timedelta(**{
        UNITS.get(m.group('unit').lower(), 'seconds'): float(m.group('val'))
        for m in re.finditer(r'(?P<val>\d+(\.\d+)?)(?P<unit>[smhdw]?)', s, flags=re.I)
    }).total_seconds()) for s in X['time_taken']]



def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def dictionary_to_columns(X, colmn):
    temp = X[colmn].values
    X.drop(colmn, inplace=True, axis=1)
    s = [ast.literal_eval(d)['source'] for d in temp]
    d = [ast.literal_eval(d)['destination'] for d in temp]
    X['source'] = s
    X['destination'] = d
    return X


def Date_Converter(X):
    datalist = [datetime.timestamp(datetime.strptime(d, '%d/%m/%Y')) for d in [t.replace('-', '/') for t in X['date'].values]]
    X['date'] = datalist
    temp=[int(str(p)[:-4]) for p in X['date'].values]
    X['date']=temp

    return X


def Stop_Feature(column):
    values = ["1stop", "nonstop", "2stop"]
    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                  "*", "+", ",", "-", ".", "/", ":", ";", "<",
                  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                  "`", "{", "|", "}", "~", "–"]
    for char in spec_chars:
        column = column.str.replace(char, '', regex=True)

    column = column.replace(values[0], 1, regex=True)
    column = column.replace(values[1], 0, regex=True)
    column = column.replace(values[2], 2, regex=True)
    return column



def converttomin(x):
    x=x.str.split(':').apply(lambda x: int(x[0]) * 60*60 + int(x[1])*60)
    return x


def handel_price(Y):
    return [ int(f) for f in[ t.replace(',','') for t in Y]]





