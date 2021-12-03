# -*- coding: utf-8 -*-
"""
Created on Wed Nov  17 15:56:14 2021

@author: Wanlu Zhang

Function: This file is used for model training and predictions for Q1 based on all algorithms (Two categories)
          The final comparison is based on the results of all different classification algorithms
"""
#%%
# Import libaries
import pandas as pd
import numpy as np
import matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


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

#%% Logistic Regression Classifier 
print("==========================================")      
from sklearn.linear_model import LogisticRegressionCV

Cs = [1, 10, 100, 1000]
def logRegrCVPredict(x_train, y_train, Cs, x_val):
    logcv = LogisticRegressionCV(Cs = Cs, cv = 10, max_iter = 2000, fit_intercept=True, random_state=0, solver='lbfgs')
    logcv.fit(x_train,y_train)
    print('Best Cs value: '+str(logcv.C_))
    y_pred = logcv.predict(x_val)
    return y_pred
y_pred_LRCV =  logRegrCVPredict(x_train, y_train, Cs, x_val)
mse_LRCV=mean_squared_error(y_val,y_pred_LRCV)

print("Logistic Regression Classifier")
print(classification_report(y_val,y_pred_LRCV))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_LRCV))
print("Accuracy",accuracy_score(y_val,y_pred_LRCV))   #0.9106
 
#%% Decision Tree Classifier    
print("==========================================")   
from sklearn import tree

def DecisionTreePredict(x_train, y_train, x_val):
    tree_params={'criterion':'entropy'}
    DT = tree.DecisionTreeClassifier( **tree_params )
    DT.fit(x_train,y_train)
    y_pred = DT.predict(x_val)
    return y_pred
y_pred_DT =  DecisionTreePredict(x_train, y_train, x_val)
mse_DT=mean_squared_error(y_val,y_pred_DT)

print("Decision Tree")
print(classification_report(y_val,y_pred_DT))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_DT))
print("Accuracy",accuracy_score(y_val,y_pred_DT))   #0.92133
 
#%% GBDT(Gradient Boosting Decision Tree) Classifier    
print("==========================================")   
from sklearn.ensemble import GradientBoostingClassifier

def GBDecisionTreePredict(x_train, y_train, x_val):
    GBDT = GradientBoostingClassifier(n_estimators=100)
    GBDT.fit(x_train,y_train)
    y_pred = GBDT.predict(x_val)
    return y_pred
y_pred_GBDT =  GBDecisionTreePredict(x_train, y_train, x_val)
mse_GBDT=mean_squared_error(y_val,y_pred_GBDT)

print("Gradient Boosting Decision Tree")
print(classification_report(y_val,y_pred_GBDT))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_GBDT))
print("Accuracy",accuracy_score(y_val,y_pred_GBDT))   #0.912

#%% Random Forest
print("==========================================")   
from sklearn.ensemble import RandomForestClassifier

def RandomForestPredict(x_train, y_train, x_val):
    RF = RandomForestClassifier(n_estimators=50,random_state=534067695)
    RF.fit(x_train,y_train)
    print('Best random state: '+str(RF.estimators_[0]))    
    y_pred = RF.predict(x_val)
    return y_pred
y_pred_RF =  RandomForestPredict(x_train, y_train, x_val)
mse_RF=mean_squared_error(y_val,y_pred_RF)

print("Random Forest")
print(classification_report(y_val,y_pred_RF))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_RF))
print("Accuracy",accuracy_score(y_val,y_pred_RF))   #0.9186

#%%AdaBoost Classifier
print("==========================================")   
from sklearn.ensemble import  AdaBoostClassifier

def AdaBoostPredict (x_train, y_train, x_val):
    AB = AdaBoostClassifier(n_estimators = 300)
    AB.fit(x_train,y_train, sample_weight = None)
    y_pred = AB.predict(x_val)
    return y_pred
y_pred_AB =  AdaBoostPredict(x_train, y_train, x_val)
mse_AB=mean_squared_error(y_val,y_pred_AB)

print("AdaBoost")
print(classification_report(y_val,y_pred_AB))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_AB))
print("Accuracy",accuracy_score(y_val,y_pred_AB))    #0.9173

#%% KNN
print("==========================================")   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def KNNClassifierPredict(x_train, y_train, x_val,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train) # Fit KNN model
    y_pred = neigh.predict(x_val)
    return y_pred

##
KNN_k = [x for x in range(1,round(len(x_train)**0.5)+1)]
kf=KFold(n_splits=100,random_state=0,shuffle=True)
k_candidate = []
for k in KNN_k:
    score = 0
    for train_index,valid_index in kf.split(x_train):
        y_pred = KNNClassifierPredict(x_train[train_index], y_train[train_index], x_train[valid_index],k)
        score =score + accuracy_score(y_train[valid_index],y_pred)
    avg_score = score/kf.n_splits
    k_candidate.append(avg_score)
#    print('\r'"Cross Validation Process:{0}%".format(round(k * 100 / len(KNN_k))), end="",flush=True)

k_best =k_candidate.index(max(k_candidate))+1
print('\nBest k: '+str(k_best))

y_pred_KNN=KNNClassifierPredict(x_train, y_train, x_val,k_best)
mse_cv_KNN =mean_squared_error(y_val,y_pred_KNN)

print("KNN")
print(classification_report(y_val,y_pred_KNN))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv_KNN))
print("Accuracy",accuracy_score(y_val,y_pred_KNN))    #0.9546


#%% SVM Classifier 
print("==========================================")   
from sklearn.svm import SVC

C = 1.0  # SVM regularization parameter

def SVMPredict(x_train,y_train, x_val):
#    model = SVC(kernel='linear', C=C, probability=True)              #0.908
#    model = SVC(kernel='rbf', gamma=0.7, C=C, probability=True)      #0.92
    model = SVC(kernel='poly', degree=2, C=C)       #0.9213
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    return y_pred
y_pred_SVM = SVMPredict(x_train,y_train, x_val)
mse_SVM = mean_squared_error(y_val,y_pred_SVM)

print("SVM")
print(classification_report(y_val,y_pred_SVM))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_SVM))
print("Accuracy",accuracy_score(y_val,y_pred_SVM))     #0.92133


#%% Multinomial Naive Bayes Classifier
print("==========================================")       
from sklearn.naive_bayes import MultinomialNB
alphas = [0.01, 0.1, 1, 10, 100]

def BayesPredict(x_train, y_train, alphas, x_val):
    Bayes = MultinomialNB(alpha = 0.01)
    Bayes.fit(x_train,y_train)
#    print('Best alpha value: '+str(Bayes.alpha_))    
    y_pred = Bayes.predict(x_val)
    return y_pred
y_pred_Bayes = BayesPredict(x_train,y_train, alphas, x_val)
mse_Bayes = mean_squared_error(y_val,y_pred_Bayes)

print("Multinomial Naive Bayes")
print(classification_report(y_val,y_pred_Bayes))
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_Bayes))
print("Accuracy",accuracy_score(y_val,y_pred_Bayes))  #0.83733


