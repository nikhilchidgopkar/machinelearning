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


from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('# Clusters')
plt.ylabel('Cost')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,random_state=0)
y_clusters = kmeans.fit_predict(X)
    
plt.scatter(X[:,0], X[:,1])
plt.show();

plt.scatter(X[y_clusters == 0 ,0], X[y_clusters == 0,1], s=100, c= 'red', label='Careful')
plt.scatter(X[y_clusters == 1 ,0], X[y_clusters == 1,1], s=100, c= 'black',label='Standard 1')
plt.scatter(X[y_clusters == 2 ,0], X[y_clusters == 2,1], s=100, c= 'green', label = 'Target1: Rich')
plt.scatter(X[y_clusters == 3 ,0], X[y_clusters == 3,1], s=100, c= 'pink', label = 'Target 2: Careless')
plt.scatter(X[y_clusters == 4 ,0], X[y_clusters == 4,1], s=100, c= 'cyan', label= 'Standard')
plt.scatter(kmeans.cluster_centers_[: ,0], kmeans.cluster_centers_[:,1], s=300, c= 'yellow', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()    


