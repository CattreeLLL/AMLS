# -*- coding: utf-8 -*-
"""
Created on Wed Dec 1 15:56:14 2021

@author: Wanlu Zhang

Function: This file is used for model training and predictions for Task A, including all
          algorithms used.
          
"""

# Import External Libraries
import pandas as pd                                                  # pandas for label reading
import numpy as np                                                   # numpy for basic array operation
import matplotlib.pyplot as plt                                      # plt for figure plotting
from sklearn.linear_model import LogisticRegressionCV                # Logistic Regression Classifier 
from sklearn.tree import DecisionTreeClassifier                      # Decision Tree Classifier
from sklearn.ensemble import GradientBoostingClassifier              # GBDT(Gradient Boosting Decision Tree) Classifier
from sklearn.ensemble import RandomForestClassifier                  # Random Forest
from sklearn.ensemble import  AdaBoostClassifier                     # AdaBoost Classifier
from sklearn.neighbors import KNeighborsClassifier                   # KNN Classifier
from sklearn.model_selection import KFold                            # KFold
from sklearn.svm import SVC                                          # SVM Classifier 
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score     # classification report and accuracy display
from sklearn.preprocessing import MinMaxScaler                       # data scaling


#%% Read PCA down_sampled data and labels

x_train = np.array(pd.read_csv('./dataset/image/x_train_pca.csv', header=None))           # MRI images for module training
y_train = np.array(pd.read_csv('./dataset/image/y_train.csv', header=None)).ravel()    # labels for module training
x_val = np.array(pd.read_csv('./dataset/image/x_val_pca.csv', header=None))               # MRI images for module validation
y_val = np.array(pd.read_csv('./dataset/image/y_val.csv', header=None)).ravel()        # labels for module training  
x_test = np.array(pd.read_csv('./dataset/image/x_test_pca.csv', header=None)) 
y_test = np.array(pd.read_csv('./dataset/image/y_val.csv', header=None)).ravel()  

#%% Scaling training and validation data

scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
x_train = scaler.fit_transform(x_train)                             # scales training set
x_val = scaler.transform(x_val)                                     # scales validation set

#%% 
# ============================================================================== #
#                                                                                #
#                           Logistic Regression Classifier                       #
#                                                                                #
# ============================================================================== #

print("==========================================")      

Cs = [1, 10, 100, 1000]                                             # cross-validation
# Function defination
def logRegrCVPredict(x_train, y_train, Cs, x_val):
    logcv = LogisticRegressionCV(Cs = Cs, cv = 10, max_iter = 4000, fit_intercept=True, random_state=0, solver='lbfgs')
    logcv.fit(x_train,y_train)                                      # model fitting
    print('Best Cs value: '+str(logcv.C_))
    y_pred = logcv.predict(x_val)                                   # predict y values based on x_val
    return y_pred

# Data prediction for validation set
y_pred_LRCV =  logRegrCVPredict(x_train, y_train, Cs, x_val)        # call function to predict y values based on x_val
mse_LRCV = mean_squared_error(y_val,y_pred_LRCV)                      # mean squared error

print("Logistic Regression Classifier (validation set)")
print(classification_report(y_val,y_pred_LRCV))                     # display classification report
print('Mean Squared Error (MSE) on validation set (built-in cross-validation): '+str(mse_LRCV))
print("Accuracy",accuracy_score(y_val,y_pred_LRCV))                 # 0.9146
print("==========================================") 

# Data prediction for test set
y_pred_test_LRCV =  logRegrCVPredict(x_train, y_train, Cs, x_test)  # call function to predict y values based on x_test
mse_test_LRCV = mean_squared_error(y_test,y_pred_test_LRCV)         # mean squared error

print("Logistic Regression Classifier (test set)")
print(classification_report(y_test,y_pred_test_LRCV))               # display classification report
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_test_LRCV))
print("Accuracy",accuracy_score(y_test,y_pred_test_LRCV))           # 0.9146




#%% 
# ============================================================================== #
#                                                                                #
#                              Decision Tree Classifier                          #
#                                                                                #
# ============================================================================== #
   
print("==========================================")   

def DecisionTreePredict(x_train, y_train, x_val):
    tree_params={'criterion':'entropy'}                             # tree parameters
    DT = DecisionTreeClassifier( **tree_params )
    DT.fit(x_train,y_train)                                         # model fitting
    y_pred = DT.predict(x_val)                                      # predict y values based on x_val
#    plot_tree(DT,filled=Ture,feature_names=x_train(), class_names=str(y_train))    # tree visualization
    return y_pred

# Data prediction for validation set
y_pred_DT =  DecisionTreePredict(x_train, y_train, x_val)           # call function to predict y values based on x_val
mse_DT=mean_squared_error(y_val,y_pred_DT)                          # mean squared error

print("Decision Tree (validation set)")
print(classification_report(y_val,y_pred_DT))                       # display classification report
print('Mean Squared Error (MSE) on validation set: '+str(mse_DT))
print("Accuracy",accuracy_score(y_val,y_pred_DT))                   # 0.92133
print("==========================================") 

# Data prediction for test set
y_pred_test_DT =  DecisionTreePredict(x_train, y_train, x_test)     # call function to predict y values based on x_val
mse_test_DT=mean_squared_error(y_test,y_pred_test_DT)               # mean squared error

print("Decision Tree (test set)")
print(classification_report(y_test,y_pred_test_DT) )                # display classification report
print('Mean Squared Error (MSE) on test set: '+str(mse_test_DT))
print("Accuracy",accuracy_score(y_test,y_pred_test_DT) )            # 0.92133



#%% 
# ============================================================================== #
#                                                                                #
#                  GBDT(Gradient Boosting Decision Tree) Classifier              #
#                                                                                #
# ============================================================================== #
 
print("==========================================")   

def GBDecisionTreePredict(x_train, y_train, x_val):
    GBDT = GradientBoostingClassifier(n_estimators=50)               # built classifier
    GBDT.fit(x_train,y_train)                                        # model fitting
    y_pred = GBDT.predict(x_val)                                     # predict y values based on x_val
    return y_pred

# Data prediction for validation set
y_pred_GBDT =  GBDecisionTreePredict(x_train, y_train, x_val)        # call function to predict y values based on x_val
mse_GBDT=mean_squared_error(y_val,y_pred_GBDT)                       # mean squared error

print("Gradient Boosting Decision Tree (validation set)")
print(classification_report(y_val,y_pred_GBDT))                      # display classification report               
print('Mean Squared Error (MSE) on validation set: '+str(mse_GBDT))
print("Accuracy",accuracy_score(y_val,y_pred_GBDT))                  # 0.912
print("==========================================") 

# Data prediction for test set
y_pred_test_GBDT =  GBDecisionTreePredict(x_train, y_train, x_test)  # call function to predict y values based on x_val
mse_test_GBDT=mean_squared_error(y_test,y_pred_test_GBDT)            # mean squared error

print("Gradient Boosting Decision Tree (test set)")
print(classification_report(y_test,y_pred_test_GBDT) )               # display classification report               
print('Mean Squared Error (MSE) on test set: '+str(mse_test_GBDT))
print("Accuracy",accuracy_score(y_test,y_pred_test_GBDT) )           # 0.912



#%% 
# ============================================================================== #
#                                                                                #
#                                   Random Forest                                #
#                                                                                #
# ============================================================================== #

print("==========================================")   

def RandomForestPredict(x_train, y_train, x_val):
    RF = RandomForestClassifier(n_estimators=50,random_state=5)     # built classifier
    RF.fit(x_train,y_train)                                         # model fitting
    print('Best random state: '+str(RF.estimators_[0]))    
    y_pred = RF.predict(x_val)                                      # predict y values based on x_val
    return y_pred

# Data prediction for validation set
y_pred_RF =  RandomForestPredict(x_train, y_train, x_val)           # call function to predict y values based on x_val
mse_RF=mean_squared_error(y_val,y_pred_RF)                          # mean squared error

print("Random Forest (validation set)")
print(classification_report(y_val,y_pred_RF))                       # display classification report               
print('Mean Squared Error (MSE) on validation set: '+str(mse_RF))
print("Accuracy",accuracy_score(y_val,y_pred_RF))                   # 0.9186
print("==========================================") 

# Data prediction for test set
y_pred_test_RF =  RandomForestPredict(x_train, y_train, x_test)      # call function to predict y values based on x_val
mse_test_RF=mean_squared_error(y_test,y_pred_test_RF)                # mean squared error

print("Random Forest (test set)")
print(classification_report(y_test,y_pred_test_RF) )                 # display classification report               
print('Mean Squared Error (MSE) on test set: '+str(mse_test_RF))
print("Accuracy",accuracy_score(y_test,y_pred_test_RF) )             # 0.9186



#%% 
# ============================================================================== #
#                                                                                #
#                               AdaBoost Classifier                              #
#                                                                                #
# ============================================================================== #

print("==========================================")   

def AdaBoostPredict (x_train, y_train, x_val):
    AB = AdaBoostClassifier(n_estimators = 300)                     # built classifier
    AB.fit(x_train,y_train, sample_weight = None)                   # model fitting
    y_pred = AB.predict(x_val)                                      # predict y values based on x_val
    return y_pred

# Data prediction for validation set
y_pred_AB =  AdaBoostPredict(x_train, y_train, x_val)               # call function to predict y values based on x_val
mse_AB=mean_squared_error(y_val,y_pred_AB)                          # mean squared error

print("AdaBoost (validation set)")
print(classification_report(y_val,y_pred_AB))                       # display classification report               
print('Mean Squared Error (MSE) on validation set: '+str(mse_AB))
print("Accuracy",accuracy_score(y_val,y_pred_AB))                   #0.9173
print("==========================================") 

# Data prediction for test set
y_pred_test_AB =  AdaBoostPredict(x_train, y_train, x_test)               # call function to predict y values based on x_val
mse_test_AB=mean_squared_error(y_test,y_pred_test_AB)                          # mean squared error

print("AdaBoost (test set)")
print(classification_report(y_test,y_pred_test_AB))                       # display classification report               
print('Mean Squared Error (MSE) on test set: '+str(mse_test_AB))
print("Accuracy",accuracy_score(y_test,y_pred_test_AB) )                   #0.9173




#%% 
# ============================================================================== #
#                                                                                #
#                                        KNN                                     #
#                                                                                #
# ============================================================================== #

print("==========================================")   

def KNNClassifierPredict(x_train, y_train, x_val,k):
    neigh = KNeighborsClassifier(n_neighbors=k)                      # built classifier
    neigh.fit(x_train, y_train)                                      # model fitting
    y_pred = neigh.predict(x_val)                                    # predict y values based on x_val
    return y_pred

##  K fold cross-validation
KNN_k = [x for x in range(1,round(len(x_train)**0.5)+1)]             # initial k set
kf=KFold(n_splits=10,random_state=0,shuffle=True)
k_candidate = []
for k in KNN_k:
    score = 0
    for train_index,valid_index in kf.split(x_train):
        y_pred = KNNClassifierPredict(x_train[train_index], y_train[train_index], x_train[valid_index],k)
        score = score + accuracy_score(y_train[valid_index],y_pred)
    avg_score = score/kf.n_splits                                     # obtain the average of scores
    k_candidate.append(avg_score)                                     # save into an array

k_best = k_candidate.index(max(k_candidate))+1                        # calculate the best k value
print('\nBest k: '+str(k_best))

# Data prediction for validation set
y_pred_KNN=KNNClassifierPredict(x_train, y_train, x_val,k_best)       # call function to predict y values based on x_val
mse_cv_KNN =mean_squared_error(y_val,y_pred_KNN)                      # mean squared error

print("KNN (validation set)")
print(classification_report(y_val,y_pred_KNN))                        # display classification report               
print('Mean Squared Error (MSE) on validation set (built-in cross-validation): '+str(mse_cv_KNN))
print("Accuracy",accuracy_score(y_val,y_pred_KNN))                    # 0.9546
print("==========================================")

# Data prediction for test set
y_pred_test_KNN=KNNClassifierPredict(x_train, y_train, x_test,k_best)       # call function to predict y values based on x_val
mse_cv_test_KNN =mean_squared_error(y_test,y_pred_test_KNN)                      # mean squared error

print("KNN (test set)")
print(classification_report(y_test,y_pred_test_KNN))                        # display classification report               
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv_test_KNN))
print("Accuracy",accuracy_score(y_test,y_pred_test_KNN))  

#%%  Accuracy comparison for different k value
def k_plotter(x_train, y_train, x_val, y_val, k_set):                 # plot the accuracy variation based on differnt k value      
    score_T=np.array([[0]*2 for i in range(k_set)],float)
    for k in range(1,k_set+1):
        score = accuracy_score(y_val,KNNClassifierPredict(x_train, y_train, x_val,k))
        score_T[k-1,:] = [score, k]                                   # save the accuracy score    
    return score_T

score_k = k_plotter(x_train, y_train, x_val, y_val, round(len(x_train)**0.5))  # apply the defined function to current dataset

plt.plot(score_k[:,1],score_k[:,0])                                   # plot figure
plt.xlabel('K')
plt.ylabel('Accuracy Score')


#%% 
# ============================================================================== #
#                                                                                #
#                                  SVM Classifier                                #
#                                                                                #
# ============================================================================== #

print("==========================================")   

C = 1.0                                                               # SVM regularization parameter

def SVMPredict(x_train,y_train, x_val):
#    model = SVC(kernel='linear', C=C, probability=True)              #0.908
#    model = SVC(kernel='rbf', gamma=0.7, C=C, probability=True)      #0.92
    model = SVC(kernel='poly', degree=2, C=C)                         #0.9213
    model.fit(x_train,y_train)                                        # model fitting
    y_pred = model.predict(x_val)
    return y_pred

# Data prediction for validation set
y_pred_SVM = SVMPredict(x_train,y_train, x_val)                       # predict y values based on x_val
mse_SVM = mean_squared_error(y_val,y_pred_SVM)                        # mean squared error

print("SVM (validation set)")
print(classification_report(y_val,y_pred_SVM))
print('Mean Squared Error (MSE) on validation set: '+str(mse_SVM))
print("Accuracy",accuracy_score(y_val,y_pred_SVM))                    # 0.92133
print("==========================================")

# Data prediction for test set
y_pred_test_SVM = SVMPredict(x_train,y_train, x_test)                       # predict y values based on x_val
mse_test_SVM = mean_squared_error(y_test,y_pred_test_SVM)                        # mean squared error

print("SVM (test set)")
print(classification_report(y_test,y_pred_test_SVM))
print('Mean Squared Error (MSE) on test set: '+str(mse_test_SVM))
print("Accuracy",accuracy_score(y_test,y_pred_test_SVM))                    # 0.92133

