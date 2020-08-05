# W207 Final Project: Melanoma Classification
## Maria Auslander, Tim Chen, Matthew Hui, Steven Leung

## Project Overview
This project was created in order to determine a model which classifies skin images as containing a melanoma or as indiciating a benign skin condition.
This project involved a variety of feature engineering processes and SVM models and logistic regression models were tested before ultimately deciding on using a convolutional neural network (CNN) approach.

## Description of Files

### cnn_efn_svm_logistic.ipynb
This is our primary notebook containing our preferred CNN image classification model. This notebook also includes basic suport vector machine (SVM) and logistic regression implementations. Running this notebook requires access to our teams shared google drive which contains image data used in modelling.

### cnn_barebone.ipynb
This file contains a basic implementation of a CNN model. Running this notebook requires access to our teams shared google drive which contains image data used in modelling. 

### SVM_models.ipynb
This notebook is used to configure and train data using multiple SVM methods and kernels. The models are implemented with the assistance of various python packages including scikit-learn and tensorflow. Ultimately this is a secondary notebook as it does not contain our preferred model, a CNN model. Running this notebook requires access to our teams shared google drive which contains image data used in modelling.
