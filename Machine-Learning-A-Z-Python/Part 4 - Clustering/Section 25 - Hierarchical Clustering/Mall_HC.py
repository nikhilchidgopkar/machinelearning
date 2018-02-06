# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 22:57:18 2016

@author: nikhilc
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\machinelearning\\Machine Learning A-Z - Python\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv')

m = len(dataset) # Number of rows
n = len(dataset.columns) # Number of columns

#Seprating Xs & ys
X = dataset.iloc[:,[3,4]].values

# Finding Dendogram
import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(X,method='ward')) 

from sklearn.cluster import AgglomerativeClustering
hc =  AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_clusters = hc.fit_predict(X)


    
plt.scatter(X[:,0], X[:,1])
plt.show();

plt.scatter(X[y_clusters == 0 ,0], X[y_clusters == 0,1], s=100, c= 'red', label='Careful')
plt.scatter(X[y_clusters == 1 ,0], X[y_clusters == 1,1], s=100, c= 'black',label='Standard 1')
plt.scatter(X[y_clusters == 2 ,0], X[y_clusters == 2,1], s=100, c= 'green', label = 'Target1: Rich')
plt.scatter(X[y_clusters == 3 ,0], X[y_clusters == 3,1], s=100, c= 'pink', label = 'Target 2: Careless')
plt.scatter(X[y_clusters == 4 ,0], X[y_clusters == 4,1], s=100, c= 'cyan', label= 'Standard')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()    


