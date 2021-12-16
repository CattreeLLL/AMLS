# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6 22:01:59 2021

@author: Wanlu Zhang

Module function: This module is used to construct a single CNN which is then employed 
                 in the following ensemble learning procedure.
"""
# Import External Libraries
import tensorflow as tf                             # tensorflow for deep learning
from tensorflow.keras import layers, models         # keras library for builting model structure 

def CNN():
    
    model = models.Sequential()                     # model initialisation
    model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 1)))           # 1st Convolutional Layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))   

    model.add(layers.Conv2D(32, (3, 3)))            # 2nd Convolutional Layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2D(32, (3, 3)))            # 3rd Convolutional Layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
   
    model.add(layers.Conv2D(32, (3, 3)))            # 4th Convolutional Layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))                  # Dropout for regularisation
         
    model.add(layers.Flatten())                     # 1st fully connected layer
    model.add(layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Activation('relu'))  
    model.add(layers.Dropout(0.5))                  # Dropout for regularisation

    model.add(layers.Dense(64))                     # 2nd fully connected layer
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(4,activation='softmax')) # output layer, 'softmax' function for classification
    
    return model