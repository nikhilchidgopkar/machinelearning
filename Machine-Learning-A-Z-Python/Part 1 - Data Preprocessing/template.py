# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:57:18 2016

@author: nikhilc
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\machinelearning\\Machine Learning A-Z - Python\\Part 1 - Data Preprocessing\\Data.csv')

m = len(dataset) # Number of rows
n = len(dataset.columns) # Number of columns

#Seprating Xs & ys
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,n-1].values

# missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])


# Categorical Variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# string -> number -> split into diffrent column 
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()


encoder_y = LabelEncoder()
y= encoder_y.fit_transform(y)

# Split Training & Test Data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2, random_state=0)



# feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

