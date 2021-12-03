# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:39:05 2021

@author: Wanlu
"""

#%%
# Import libaries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

#%%
# Read data
x_train = np.array(pd.read_csv('./x_train_pca.csv', header=None))       # MRI images for module training
y_train = np.array(pd.read_csv('./y_train_Q1.csv', header=None)).ravel()      # MRI images for module validation
x_val = np.array(pd.read_csv('./x_val_pca.csv', header=None))
y_val = np.array(pd.read_csv('./y_val_Q1.csv', header=None)).ravel()   

#%%
# Pre-process data
scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

#%%
def KNNClassifier(x_train, y_train, x_val,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train) # Fit KNN model


    y_pred = neigh.predict(x_val)
    return y_pred

y_pred=KNNClassifier(x_train, y_train, x_val,1)

mse_cv=mean_squared_error(y_val,y_pred)
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv))

#print(confusion_matrix(y_val, y_pred_cv))
print('Accuracy on test set: '+str(accuracy_score(y_val,y_pred)))
print(classification_report(y_val,y_pred))#text report showing the main classification metrics

#print('\r'a)