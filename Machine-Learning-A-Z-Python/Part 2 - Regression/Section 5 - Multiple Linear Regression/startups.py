# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:57:18 2017

@author: nikhilc
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\Udmey_A2Z_ML\\Machine Learning A-Z - Self\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')

m = len(dataset)
n = len(dataset.columns)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,n-1].values

# missing data

#from sklearn.preprocessing import Imputer

#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:,1:3])


# Categorical Variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# string -> number -> split into diffrent column 
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder_X = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder_X.fit_transform(X).toarray()

# Avoid Dummpy Variable Trap
X = X[:,1:] # skipping the 1st dummy varaible out of the 3 generated from above code

# Split Training & Test Data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# feature Scaling
#from sklearn.preprocessing import StandardScaler

#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.fit_transform(X_test)

#Fitting the Simple Linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict
y_test_predict = regressor.predict(X_test)

# Optimizing the model by backward elimnation
import statsmodels.formula.api as sm

# adding X0 
onesArray = np.ones((m,1)).astype(int)
X = np.append(arr=onesArray,values=X,axis=1)

X_OPT = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_ols.summary()

X_OPT = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_ols.summary()

X_OPT = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_ols.summary()

X_OPT = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_ols.summary()

X_OPT = X[:,[0,3]]
regressor_ols = sm.OLS(endog=y, exog=X_OPT).fit()
regressor_ols.summary()



