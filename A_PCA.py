"""
Created on Mon Nov 29 22:10:19 2021

@author: Wanlu Zhang

Function: This file is used for data pre-precessing (deminsion reduction) using PCA technique.
"""

# Import External Libraries
import numpy as np
from sklearn.model_selection import train_test_split       # split the training set and validation set
from sklearn.preprocessing import StandardScaler           # scale the data
from sklearn.decomposition import PCA                      # PCA function for down-sampling
# Import submodule
import Data_Preprocessing as dp                            # data reading, label reading and down sampling

# Read the training set (1D)
set = 0
d = 1
img = dp.data_reading(set,d)
label = dp.label_reading(set,d)

# Read the test set (1D)
set = 1
x_test = dp.data_reading(set,d)
y_test = dp.label_reading(set,d)

# Split the training set and validation set
x_train, x_val, y_train, y_val = train_test_split(img, label, test_size=0.25, random_state=4) 

#%% Scale the data for training, validation and test set
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_val = scaler.transform(x_val) 
x_test = scaler.transform(x_test) 

#%% Employ PCA technique for data down-sampling for the training, validation and test set
pca = PCA(n_components = 0.95)
x_train = pca.fit_transform(x_train) 
x_val = pca.transform(x_val)
x_test = pca.transform(x_test)

#%% Save the down-sampled image data for future reuse
np.savetxt('./dataset/image/x_train_pca.csv', x_train, delimiter = ',')
np.savetxt('./dataset/image/x_val_pca.csv', x_val, delimiter = ',')
np.savetxt('./dataset/image/x_test_pca.csv', x_test, delimiter = ',')

#%% Save the corresponding encoded labels for future reuse
np.savetxt('./dataset/image/y_train.csv', y_train, delimiter = ',')
np.savetxt('./dataset/image/y_val.csv', y_val, delimiter = ',')
np.savetxt('./dataset/image/y_test.csv', y_test, delimiter = ',')

















          