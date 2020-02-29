# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
'''
Part 1 - Data Prepocessing
'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\IT\\UDEMY\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Artificial_Neural_Networks\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
Part 2. Make ANN
'''
#Import Keras
import keras
from keras.models import  Sequential#initialize NN
from keras.layers import Dense# create layers
#Initialize the ANN
classifier = Sequential()
# Adding the input layers and first hidden layers
'''
Tips:Choose the number of nodes in the hidden layers as the average of the number of nodes in the input layers and output layers.
You want to use if you don't to be a artist, if you want to be artist, use techinique (perimeter tunning).
------
Input_dim is the number of nodes in the  input layers that is the number of independent variables.
why is it compulsory to add this argument (input_dim)at this stage?
It is because so far our ANN is simply initialize, we haven't created any layers yet.
and that's why it doesn't know here which nodes this hidden layers here that we are creating is expecting as inputs.
'''
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#6=(11(input)+1(output))/2
# Add the second hidden layers
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))#no need input_dim
#Add the output layers
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
'''
If output has more categories like 3,4... Need change output_dim and activation='softmax'
Softmax is actually sigmoid function but applied to a dependent variable that has more than 2 categories.
But here we have 2 categories, 2 classes so we are finewith sigmoid
'''
#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
'''
If your denpendent variable has a binary outcome, then this logarithmic loss function is called binary_crossentropy.
Metrics is a standard that you choose to evaluate your model and typically we use accuracy
'''
#Fitting the ANN to the training set
classifier.fit(X_train, y_train,batch_size=10, epochs=100)
'''
Part 3. Making the prediction and evaluating the model
'''
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)