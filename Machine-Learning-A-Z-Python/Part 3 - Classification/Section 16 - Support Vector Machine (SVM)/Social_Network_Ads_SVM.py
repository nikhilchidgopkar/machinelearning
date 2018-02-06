# -*- coding: utf-8 -*-
"""
@author: nikhilc
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data

dataset = pd.read_csv('C:\\Users\\nikhilc\\Google Drive\\_Data Science\\Udmey_A2Z_ML\\Machine Learning A-Z - Self\\Part 3 - Classification\\Section 14 - Logistic Regression\\Social_Network_Ads.csv')

m = len(dataset)
n = len(dataset.columns)

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,n-1].values


# Split Training & Test Data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)


# feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Define the Classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred);

def plotLogisticRegression(X_set, y_set, classifier):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.show()
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    return ;

# Visual represntaion
plotLogisticRegression(X_train, y_train,classifier)



    
'''    
    X1 = np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01)
    X2 = np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01)
    X1, X2 = np.meshgrid(X1, X2)
    temp = np.array([X1, X2]).T
    X3 = classifier.predict(np.array([X1, X2]) )
    plt.contourf()
    return X1, X2;
'''






#def sayHello(msg="Hello"):
 #   print (msg)
  #  return msg ;
    
