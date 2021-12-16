# -*- coding: utf-8 -*-
"""
Created on Thu Dec 9 19:32:10 2021

@author: Wanlu Zhang

Module function: This module is the main module which calls the above sub-modules 
                 to train the models, ensembles the results and plots the figures 
                 for visualisation. 
"""
# Import External Libraries
import time                                             # display time used
import tensorflow as tf                                 # tensorflow for deep learning
import numpy as np                                      # numpy for basic array operation
import matplotlib.pyplot as plt                         # plt for figure plotting
from sklearn.model_selection import train_test_split    # train_test_split for splitting the training and validation set
from tensorflow.keras import callbacks                  # callbacks for model fitting

# Import External Modules
import B_CNN_Block as blk                               # single CNN block
import Data_Preprocessing as dp                         # data reading, label reading and down sampling
import B_Ensemble_Method as em                          # averaging and mojority voting ensemble

# image reading and label reading
set = 0                                                 # for trainning set
d = 2                                                   # 2D images
img = dp.data_reading(set,d)                            # reading 2D images
img = dp.down_sampling(img,64)                          # down sampling 2D images
label = dp.label_reading(set,d)                         # reading corresponding labels

#%% train / validation split
                                                        # split images and labels into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(img, label, test_size=0.25, random_state=4)
x_train, x_val = x_train / 255.0, x_val / 255.0         # normalisation

#%% model fitting
# initialisation
ensemble = 20                                           # total number of ensembled CNNs is 10
model_trained = list()

# individual model compile and fitting
for i in range(ensemble):
    print('Model No. {}'.format(i+1))
    
    model = blk.CNN()                                   # call CNN function in submodule B_CNN_Block
    model.compile(optimizer= 'adam',                    # training optimizer
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),         # loss function
              metrics=['accuracy'])                     # metrics
    
    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,         # stopping criterion
                                                   patience=40, verbose=0, mode='max',
                                                   baseline=None, restore_best_weights=True)
    
    history = model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val),               # compiled model fits the training set and labels
              callbacks=[early_stopping])  
    
    model_trained.append(model)                         # ensemble models
#    model.summary()
#%% data prediction for validation set
pred_val = list()                                       # array initialisation

for model in model_trained:
    pred_val.append(model.predict(x_val))               # data prediction

pred_val = np.array(pred_val)                           # array transformation

result_avg = em.avg_result(pred_val,y_val)              # average ensemble
result_vote = em.vote_result(pred_val,y_val)            # majority voting emsemble

#%% data reading for test set
set = 1                                                 # for test set
d = 2                                                   # 2D images
x_test = dp.data_reading(set,d)                         # test set reading
x_test = dp.down_sampling(x_test,64)                    # image down-sampling
y_test = dp.label_reading(set,d)                        # label reading
x_test = x_test / 255.0                                 # normalisation
#%% data prediction for test set
pred_test = list()                                      # array initialisation

start=time.process_time()
for model in model_trained:
    pred_test.append(model.predict(x_test))             # data prediction
end=time.process_time()
print("time is ",end-start)

pred_test = np.array(pred_test)                         # array transformation

result_avg = em.avg_result(pred_test,y_test)            # average ensemble
result_vote = em.vote_result(pred_test,y_test)          # majority voting emsemble

#%%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()










