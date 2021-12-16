# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6 22:48:47 2021

@author: Wanlu Zhang

Module function: This module is used to ensemble results from all 10 CNNs. 
"""
# Import External Libraries
import numpy as np                                              # numpy for basic array transformation
from sklearn.metrics import classification_report               # output a report for classification results

# ============================= Averaging Ensemble ============================= #
#     This fuction is used to averages the numerical output of all ensembled     #
#                 CNN and then chooses the final classification type.            #
# ============================================================================== #

def avg_result(pred,y_test):                                    # parameter 'pred' is the predicted result, 'y_test' is the accurate answer 

    mean = np.mean(pred, axis=0)                                # get the average numerical value of predicted results
    result = np.argmax(mean, axis=1)
    
    print(classification_report(y_test,result))                 # print the classification report 
    
    return result

def vote_result(pred,y_test):                                   # parameter 'pred' is the predicted result, 'y_test' is the accurate answer
    
    # list initialisation
    result = list()

    pred = np.argmax(pred, axis=2)                              # get the output indexes of predicted results
    for i in range(len(y_test)):   
        result.append(np.argmax(np.bincount(pred[:,i])))        # get the majority voting numerical value of predicted results

    print(classification_report(y_test,result))                 # print the classification report
    
    return result
    