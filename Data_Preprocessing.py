# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6 22:04:58 2021

@author: Wanlu Zhang

Module function: This module is used to implement the data reading and pre-processing 
                 procedure to reduce the image data dimension by simple down-sampling 
                 approach. 
"""
# Import External Libraries
import numpy as np          # numpy for basic array transformation
import pandas as pd         # pandas for label(csv file) reading
import tensorflow as tf     # tensorflow for deep learning
from PIL import Image       # Image for opening the MRI images

# ================================ Data Reading ================================ #
#     This fuction is used to load the images into the Variable Manager for      #
#       subsequent calculations. Here are four types of reading outcomes,        #
#     depending on the selection of training set or test set and the target      #
#                                 data dimension.                                #
# ============================================================================== #

def data_reading(set, d):               #set = 0, read training set; set = 1, read test set
                                        #d = 1, flatten 2D images into 1D; d = 2, read 2D images

    # array initialisation
    image_name= []                      # names of the target images
    img = []                            # target images
    
    # data reading based on input parameters 'set' and 'd'
    if set <= 0:                        # read training set

        for i in range(0,3000):         # training set contains 3000 images
            if i <= 9:
                image_name.append('./dataset/image/IMAGE_000{}.jpg'.format(i))      # read images from index 0 to 9
            elif i <= 99:
                image_name.append('./dataset/image/IMAGE_00{}.jpg'.format(i))       # read images from index 10 to 99   
            elif i <= 999:
                image_name.append('./dataset/image/IMAGE_0{}.jpg'.format(i))        # read images from index 100 to 999
            elif i <= 2999:
                image_name.append('./dataset/image/IMAGE_{}.jpg'.format(i))         # read images from index 1000 to 2999
        
        if d <= 1:                      # d = 1, flatten 2D images into 1D
            for i in image_name:        # traverse all images
                img.append(np.array(Image.open(i).convert('L'),'f').flatten())      # open and save gray and flattened 1D images
                print('\r' "Image reading: {}%".format(round(image_name.index(i)*100/3000)), end="")
        
        elif d > 1:                     # d = 2, read 2D images 
            for i in image_name:        # traverse all images
                img.append(np.array(Image.open(i).convert('L'),'f'))                # open and save gray 2D images
                print('\r' "Image reading: {}%".format(round(image_name.index(i)*100/3000)), end="")
     
    elif set > 0:                       # read test set
         
        for i in range(0,200):          # test set contains 200 images
            if i <= 9:
                image_name.append('./test/image/IMAGE_000{}.jpg'.format(i))         # read images from index 0 to 9
            elif i <= 99:
                image_name.append('./test/image/IMAGE_00{}.jpg'.format(i))          # read images from index 10 to 99 
            elif i <= 199:
                image_name.append('./test/image/IMAGE_0{}.jpg'.format(i))           # read images from index 100 to 199         
                 
        if d <= 1:                      # d = 1, flatten 2D images into 1D
            for i in image_name:        # traverse all images
                img.append(np.array(Image.open(i).convert('L'),'f').flatten())      # open and save gray and flattened 1D images
                print('\r' "Image reading: {}%".format(round(image_name.index(i)*100/200)), end="")

        elif d > 1:                     # d = 2, read 2D images 
            for i in image_name:        # traverse all images
                img.append(np.array(Image.open(i).convert('L'),'f'))                # open and save gray 2D images
                print('\r' "Image reading: {}%".format(round(image_name.index(i)*100/200)), end="")
  
    # data reshape
    img = np.array(img)                 # transfer img into an array form
    if d > 1:                           # if for 2D images
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1)              # reshape all images
    
    print('\n')
    return img
# =============================================================================



# ================================ Label Reading ================================ #
#      This fuction is used to load the labels into the Variable Manager for      #
#   subsequent calculations. Here are two types of reading outcomes, depending    #
#     depending on the selection of Task A (tumor diagnosis) and Task B (tumor    #
#                                  classification).                               #
# =============================================================================== #

def label_reading(set,d):               # et = 0, read training labels; set = 1, read test labels
                                        # d = 1, Task A (tumor diagnosis); d = 2, Task B (tumor classification)
    
    # label reading based on input parameters 'set' and 'd'
    if set <= 0:                        # read training labels  
        label = pd.read_csv('./dataset/label.csv')
        
    elif set > 0:                       # read test labels
        label = pd.read_csv('./test/label.csv')
        
    label = label['label']
    
    # label encoding
    i = 0
    if d <= 1:                          # d = 1, Task A (tumor diagnosis): 'no tumor' is encoded as 0; other tumors are encoded as 1
        for i in range(0, len(label)):
            if label[i]== 'no_tumor':
                label[i] = 0
            elif label[i] == 'meningioma_tumor':
                label[i] = 1
            elif label[i] == 'glioma_tumor':
                label[i] = 1
            elif label[i] == 'pituitary_tumor':
                label[i] = 1
            print('\r'"Label reading:{0}%".format(round((i + 1) * 100 / len(label))), end="")
    
    elif d > 1:                         # d = 2, Task B (tumor classification): 'no tumor' is encoded as 0; other tumors are encoded as 1 to 3 correspondingly
        for i in range(0, len(label)):
            if label[i]== 'no_tumor':
                label[i] = 0
            elif label[i] == 'meningioma_tumor':
                label[i] = 1
            elif label[i] == 'glioma_tumor':
                label[i] = 2
            elif label[i] == 'pituitary_tumor':
                label[i] = 3
            print('\r'"Label reading:{0}%".format(round((i + 1) * 100 / len(label))), end="")        
    
    # data reshape
    label = np.array(label, int)        
    label = label.reshape(len(label), 1) 
    
    print('\n')
    return label
# ===============================================================================



# ================================ Down Samplinging ============================= #
#     This fuction is used for the down-sample procedure applied to 2D images     #
#                                      in Task B                                  #
# =============================================================================== #

def down_sampling(images, size):        # 'images' is the target image array to be down-sampled;'size' is the target size

    # array initialisation
    images_d = []
    
    # down sampling procedure
    for i in range(len(images)):        # traverse all images
        images_d.append(np.array(tf.image.resize(images[i], [size, size])))         # resize images (down-sampling)
        print('\r'"Down sampling:{0}%".format(round((i + 1) * 100 / len(images))), end="")

    # array form transformation
    images_d = np.array(images_d)
    
    print('\n')
    return images_d
# =============================================================================
