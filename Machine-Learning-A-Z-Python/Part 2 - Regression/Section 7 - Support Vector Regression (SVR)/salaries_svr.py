# -*- coding: utf-8 -*-

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\Udmey_A2Z_ML\\Machine Learning A-Z - Self\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)\\Position_Salaries.csv')

m = len(dataset)
n = len(dataset.columns)

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,n-1].values

#Feature Scaling is needed

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


#smoother curves
X_grid = np.arange(min(X),max(X)+0.1,0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')


y_grid_predict = regressor.predict(X_grid)
#plt.scatter(X,y_poly_predict,color='green')
plt.plot(X_grid,y_grid_predict,color='black')

plt.show()

#Predicting if bluff or not
input = 6.5
inputArray = np.array([[input]])
transformedX = sc_X.transform(inputArray)
transformedY = regressor.predict(transformedX)
y_predict = sc_y.inverse_transform(transformedY)

#bluffOrNot = regressor.predict(6.5)




