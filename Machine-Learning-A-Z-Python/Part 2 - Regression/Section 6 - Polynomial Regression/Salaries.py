# -*- coding: utf-8 -*-


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\Udmey_A2Z_ML\\Machine Learning A-Z - Self\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv')

m = len(dataset)
n = len(dataset.columns)

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,n-1].values

#Linear regression for comparsion 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

y_predict = regressor.predict(X)
plt.scatter(X,y, color='red')
plt.scatter(X,y_predict,color='yellow')
plt.plot(X,y_predict)

#plt.scatter(X_test,y_test,color='yellow')
#plt.scatter(X_test,y_test_predict,color='green')

plt.show()

#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polynomialFeatures = PolynomialFeatures(degree=4) # Degree 2,3,4. 4 - Perfect fit
X_Poly = polynomialFeatures.fit_transform(X)

regressor.fit(X_Poly,y)

y_poly_predict = regressor.predict(X_Poly)
#plt.scatter(X,y_poly_predict,color='green')
plt.plot(X,y_poly_predict,color='green')

plt.show()

#smoother curves
X_grid = np.arange(  min(X),max(X)+0.1,0.1)
X_grid = X_grid.reshape(len(X_grid),1)

X_Poly_grid = polynomialFeatures.fit_transform(X_grid)


y_poly_grid_predict = regressor.predict(X_Poly_grid)
#plt.scatter(X,y_poly_predict,color='green')
plt.plot(X_grid,y_poly_grid_predict,color='black')

plt.show()



bluffOrNot = regressor.predict(polynomialFeatures.fit_transform(6.5))




