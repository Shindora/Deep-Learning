# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:51:44 2020

@author: TuanVi
"""

#Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train=pd.read_csv('Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values
#Feature scaling
from sklearn.preprocessing import MinMaxScaler #normalize
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#Creating a data structure with 60 timesteps and 1 output
'''
60 timesteps means that at each time T,RNN is going to look at the 60 stock prices before time T
and time T based on trends it is capturing during these 60 previous timesteps and it will try to predict the next output.
'''
X_train,y_train=[],[]
for i in range (60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
#Reshaping
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
'''
Structure in  RNN: number ofz stock prices,timesteps,number of indicators (help predictions)
'''
#Part 2- Building RNN
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

regressor=Sequential()
"""
LSTM has 3 important arg: number of units(numbers of LSTM cells),return sequences,input shape
"""
#first LSTM layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#second LSTM layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#third LSTM layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
#Output
regressor.add(Dense(units=1))
#Compile RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
regressor.summary()
#Part 3- Predict and visualising

dataset_test=pd.read_csv("Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")
real_stock_prices=dataset_test.iloc[:,1:2]
#Prediction
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):#testset has only 20 days
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#Visualising
plt.plot(real_stock_prices,color='r',label='Real GG stock prices')
plt.plot(predicted_stock_price,color='b',label='Predict GG stock prices')
plt.title("GG stock prices prediction")
plt.xlabel('Time')
plt.ylabel('GG Stock price')
plt.legend()
plt.show()
#Evaluate
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_prices, predicted_stock_price))