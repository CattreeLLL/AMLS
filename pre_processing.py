"""
Created on Mon Nov 29 22:10:19 2021

@author: Wanlu Zhang

Function: This file is used for data pre-precessing (deminsion reduction) 
"""

#%%
# import libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
#%%

a = np.array(Image.open('./IMAGE_0000.jpg').convert('L'),'f')
# plt.imshow(a, cmap = 'gray')

#%%
image_name= []
for i in range(0,3000):
    if i <= 9:
        image_name.append('./IMAGE_000{}.jpg'.format(i))
    elif i <= 99:
        image_name.append('./IMAGE_00{}.jpg'.format(i))
    elif i <= 999:
        image_name.append('./IMAGE_0{}.jpg'.format(i))
    elif i <= 2999:
        image_name.append('./IMAGE_{}.jpg'.format(i))
    
#%% 
img = []   
for i in image_name:
    img.append(np.array(Image.open(i).convert('L'),'f').flatten())
    print('zwl: {}%'.format(image_name.index(i)*100/3000))
    
img = np.array(img)
#%%
label = pd.read_csv('./label.csv')
label = label['label']
#%%
i = 0
for i in range(0, len(label)):
    if label[i]== 'no_tumor':
        label[i] = 0
    elif label[i] == 'meningioma_tumor':
        label[i] = 1
    elif label[i] == 'glioma_tumor':
        label[i] = 2
    elif label[i] == 'pituitary_tumor':
        label[i] = 3
        
#%%
x_train, x_val, y_train, y_val = train_test_split(img, label, test_size=0.25, random_state=4) 

#%% 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_val = scaler.transform(x_val) 

#%%

pca = PCA(n_components = 0.95)
x_train = pca.fit_transform(x_train) 
x_val = pca.transform(x_val)

#%% 
np.savetxt('x_train_pca.csv', x_train, delimiter = ',')
np.savetxt('x_val_pca.csv', x_val, delimiter = ',')

#%%
np.savetxt('y_train.csv', y_train, delimiter = ',')
np.savetxt('y_val.csv', y_val, delimiter = ',')

#%%
# Divide label into two sets to identify whether there is a tumor in the MRI images
        
# Division for y_train
# i = 0
# for i in range(0, len(y_train)):
#     if y_train[i]== 0:
#         y_train[i] = 0
#     elif y_train[i] == 1:      #Tumor exists (all three types)
#         y_train[i] = 1
#     elif y_train[i] == 2:      #Tumor exists (all three types)
#         y_train[i] =1
#     elif y_train[i] == 3:      #Tumor exists (all three types)
#         y_train[i] =1

# Division for y_val
# i = 0
# for i in range(0, len(y_val)):
#     if y_val[i]== 0:
#         y_val[i] = 0
#     elif y_val[i] == 1:      #Tumor exists (all three types)
#         y_val[i] = 1
#     elif y_val[i] == 2:      #Tumor exists (all three types)
#         y_val[i] =1
#     elif y_val[i] == 3:      #Tumor exists (all three types)
#         y_val[i] =1
        
# #%% Save data for model training for question 1
# np.savetxt('y_train_Q1.csv', y_train, delimiter = ',')
# np.savetxt('y_val_Q1.csv', y_val, delimiter = ',')


















          