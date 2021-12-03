# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:15:25 2021

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
tree_params={'criterion':'entropy'}
clf = tree.DecisionTreeClassifier( **tree_params )

#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(x_train,y_train)

#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(x_val)

print(f'Test feature {x_val[0]}\n True class {y_val[0]}\n predict class {y_pred[0]}')

mse_cv=mean_squared_error(y_val,y_pred)
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv))

#Use accuracy metric from sklearn.metrics library
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(x_train)))
print('Accuracy Score on validation data: ', accuracy_score(y_true=y_val, y_pred=y_pred))
print(classification_report(y_val,y_pred))


#%%
# def visualise_tree(tree_to_print):
#     plt.figure()
#     fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
#     tree.plot_tree(tree_to_print,
#                feature_names = x_train,
#                class_names = y_train, 
#                filled = True,
#               rounded = True);
#     plt.show()
#     visualise_tree(clf)