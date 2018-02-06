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

dataset = pd.read_csv('C:\\Users\\nikhilc\Google Drive\\_Data Science\\machinelearning\\Machine Learning A-Z - Python\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv')

m = len(dataset)
n = len(dataset.columns)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,n-1].values

#Splitting the data in train & test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Fitting the Simple Linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict
y_test_predict = regressor.predict(X_test)

#Plot the results
y_train_predict = regressor.predict(X_train)
plt.scatter(X_train,y_train, color='red')
plt.scatter(X_train,y_train_predict,color='blue')
plt.plot(X_train,y_train_predict)

plt.scatter(X_test,y_test,color='yellow')
plt.scatter(X_test,y_test_predict,color='green')

plt.show()