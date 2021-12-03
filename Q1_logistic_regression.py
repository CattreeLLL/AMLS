# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:10:19 2021

@author: Wanlu

Function: This file is used for model training (logistic regression)
"""
#%%
# Import libaries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, mean_squared_error
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
# generate a range of alpha values and put them in a numpy array
Cs = [1, 10, 100, 1000]


def logRegrCVPredict(x_train, y_train, Cs, x_val):

    logcv = LogisticRegressionCV(Cs = Cs, cv = 10, max_iter = 1e3, fit_intercept=True, random_state=0, solver='lbfgs')
    logcv.fit(x_train,y_train)
    print('Best Cs value: '+str(logcv.C_))
    y_pred_cv = logcv.predict(x_val)
    return y_pred_cv

y_pred_cv =  logRegrCVPredict(x_train, y_train, Cs, x_val)
mse_cv=mean_squared_error(y_val,y_pred_cv)
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv))

#print(confusion_matrix(y_val, y_pred_cv))
print('Accuracy on test set: '+str(accuracy_score(y_val,y_pred_cv)))
print(classification_report(y_val,y_pred_cv))#text report showing the main classification metrics

