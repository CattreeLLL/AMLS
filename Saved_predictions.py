# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:06:32 2021

@author: Wanlu Zhang

Module function: This module is used to load the saved predictions to directly 
                 review the classification reports.
"""
import numpy as np
import pandas as pd
import B_Ensemble_Method as em                          # averaging and mojority voting ensemble
from sklearn.metrics import classification_report       # output a report for classification results

# load saved predictions
y_test_A = np.load('./y_test_A.npy')
y_pred_A = np.load('./y_pred_A.npy')

y_test_B = np.load('./y_test_B.npy')
y_pred_B = np.load('./y_pred_B.npy')


#%% Task A
print(classification_report(y_test_A,y_pred_A))          # print classification report 

#%% Task B
result_avg = em.avg_result(y_pred_B,y_test_B)            # average ensemble
result_vote = em.vote_result(y_pred_B,y_test_B)          # majority voting emsemble