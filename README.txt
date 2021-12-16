---------------------------------------
                 Introduction
---------------------------------------

This project is used for MRI brain tumor diagnosis and classification based on machine learning classification method. 

Task A: brain tumor diagnosis to determine whether there is a brain tumor or not (binary classification)

Task B: brain tumor classification to determine the brain tumor type (four-class classification)

---------------------------------------
        Organization of the files
---------------------------------------

Data_Preprocessing.py :  This module is shared by Task A and B, consisting of three functions described as follows.
                                       
                                        1. data_reading(set, d): Read 2D images/1D array for the training set, validation set, and test set;

		                        2. label_reading(set,d): Read labels for the training set, validation set, and test set; 

                                        3. down_sampling(images, size): Down-sample input images to a target size based on a predefined value.

Task A:

A_PCA.py : This module is used for data pre-precessing (deminsion reduction) using PCA technique. There is no need to run this file as the down-sampled 
                  
           data is already saved for reuse. The 6 saved files are located in './dataset/image', which are the data and encoded labels for the training set, 
    
           validation set, and test set.


(Main) A_Comparison.py : This module is used for model training and predictions for Task A, including all algorithms used:  Logistic Regression Classifier, Decision Tree 
 
                         Classifier, GBDT(Gradient Boosting Decision Tree) Classifier, Random Forest, AdaBoost Classifier, KNN, and SVM Classifier.


Task B:

B_CNN_Block.py : This module is used to construct a single CNN which is then employed in the following ensemble learning procedure.

B_Ensemble_Method.py : This module is used to ensemble results from all 10 CNNs (Ensemble learning applied). There are two functions described as follows.

(Main) B_Ensemble_Learning.py : This module is the main module which calls the above submodules to train the models, ensembles the results and plots the figures for visualisation.                              
                

               
Saved data opening:

Saved predictions.py : Two sets of predictions for Task A and B are saved. This module can open the saved predicted data and output the classification report.  

y_test_A.npy : Real labels for Task A, binary classification.               

y_pred_A.npy : Predicted labels for Task A, binary classification.               

y_test_B.npy : Real labels for Task B, 4-class classification.   

y_pred_B.npy : Predicted labels for Task B, 4-class classification.         


---------------------------------------
        Steps to run the code
---------------------------------------   

Direct Result Check: 

              1. make sure the saved predictions are placed at the right place. They should be placed with the file 'Saved predictions.py' 

              2. run the module 'Saved predictions.py' 

Task A:

              1. make sure all .py files are placed at the right place. They should be placed outside the folders 'test' and 'dataset' which are the dataset provided.

              2. make sure the down-sampled data files provided are right placed into './dataset/image/' (6 files: x_train_pca, x_val_pca, x_test_pca, y_train, y_val, y_test .csv)

              3. open and run 'A_Comparison.py'

Task B:

             1. make sure all .py files are placed at the right place. They should be placed outside the folders 'test' and 'dataset' which are the dataset provided.

             2. run file 'B_Ensemble_Learning.py'

---------------------------------------
        Necessary External Library
---------------------------------------  

              1. numpy              

              2. pandas

              3. tensorflow

              4. tensorflow.keras

              5. matplotlib

              6. sklearn 

              7. time

              8. PIL
