# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:45:05 2019

Use a RNN structure to predict the next value for the oil price based on the previous values.
Use the dataset provided.
Evaluate the impact of adding more stacked layers to the RNN.

Note that you have to first remove the rows without a price value
dataset_train = pd.read_csv('Oilprices.csv')
dataset_train = dataset_train[dataset_train['DCOILWTICO'] !='.']
training_set = dataset_train.iloc[:, 1:2].values

Scale the data for better convergence:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

Use the first 1000 samples for training and the rest for validation.
X_train = []
y_train = []
for i in range(60, 1000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#read and load the data
data = pd.read_csv("Oilprices.csv")
data = data[data['DCOILWTICO'] !='.']
training_set = data.iloc[:, 1:2].values

#scale the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)






# create train data
X_train = []
y_train = []
for i in range(60, 1000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
y_test = []
for i in range(1000, 1254):
    X_test.append(training_set_scaled[i-60:i,0])
    y_test.append(training_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))





# create a model.
#I test the model with 3 different stacked layer.
# I choosed the parameters in order to use less computional power as much as i can.
# the idea is to see that how the number of the stacked layers effect the model.that's why 
#the model is not predict well.
# the result:
#4 stacked layer loss = 0.0229
#5 stacked layer loss = 0.0232
#9 stacked layer loss = 0.0292
# I got similar result so when the number of the stacked layer is increased, it does not means that
# we will have a better model.

# the best model parameters:
# epochs = 100
# batch size = 32
# number of neurons:  5

#################################################################################################
#4 stacked layer loss = 0.0229
    
regressor = Sequential()

regressor.add(LSTM(units = 5, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

regressor.fit(X_train, y_train, epochs = 20, batch_size = 100) # data fits well with 100 epochs
    
predicted_price = regressor.predict(X_test)
     
#plt.plot(y_test, color = 'black', label = 'Real price')
#plt.plot(predicted_price, color = 'green', label = 'Predicted price')
#plt.title('price Prediction')
#plt.xlabel('Time')
#plt.ylabel('price Prediction')
#plt.legend()
#plt.show()
###########################################################################################


#################################################################################################
#5 stacked layer loss = 0.0232
    
regressor = Sequential()

regressor.add(LSTM(units = 5, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

regressor.fit(X_train, y_train, epochs = 20, batch_size = 100) # data fits well with 100 epochs
    
predicted_price = regressor.predict(X_test)
     
#plt.plot(y_test, color = 'black', label = 'Real price')
#plt.plot(predicted_price, color = 'green', label = 'Predicted price')
#plt.title('price Prediction')
#plt.xlabel('Time')
#plt.ylabel('price Prediction')
#plt.legend()
#plt.show()
###########################################################################################





#################################################################################################
#9 stacked layer loss = 0.0292
    
regressor = Sequential()

regressor.add(LSTM(units = 5, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 5))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 20, batch_size = 100) # data fits well with 100 epochs
    
predicted_price = regressor.predict(X_test)
     
#plt.plot(y_test, color = 'black', label = 'Real price')
#plt.plot(predicted_price, color = 'green', label = 'Predicted price')
#plt.title('price Prediction')
#plt.xlabel('Time')
#plt.ylabel('price Prediction')
#plt.legend()
#plt.show()
###########################################################################################