# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 01:38:27 2021

@author: Lenovo
"""

#%%
# Import libaries
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC

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
C = 1.0  # SVM regularization parameter

def SVM(x_train,y_train, x_val):
#    model = SVC(kernel='linear', C=C)                   #0.908
#    model = SVC(kernel='rbf', gamma=0.7, C=C)      #0.92
    model = SVC(kernel='poly', degree=2, C=C)       #0.9213
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    return y_pred
# Scikit learn library results
y_pred=SVM(x_train,y_train, x_val)
print(accuracy_score(y_val,y_pred))



