# -*- coding: utf-8 -*-

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('.\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Churn_Modelling.csv')

m = len(dataset) # Number of rows
n = len(dataset.columns) # Number of columns

#Seprating Xs & ys
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,n-1].values

# Categorical Variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# string -> number -> split into diffrent column 
labelEncoder_X_Country = LabelEncoder()
labelEncoder_X_Gender = LabelEncoder()

X[:,1] = labelEncoder_X_Country.fit_transform(X[:,1])
X[:,2] = labelEncoder_X_Gender.fit_transform(X[:,2])

oneHotEncoder_X = OneHotEncoder(categorical_features = [1])
X = oneHotEncoder_X.fit_transform(X).toarray()
X = X[:, 1:]

# Split Training & Test Data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#ANN Layer creation
import tensorflow as tf
import keras.models.Sequential
import keras.layers.Dense

