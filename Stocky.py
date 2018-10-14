# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:30:17 2018

@author: prnvb
"""

#Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
#training_set will contain the opening price. 1:2 will make it the numpy array of size (m,1) and not (m,)

#Feature scaling
"""
2 ways of feature scaling:
Standardization:
    x = x - mean/standard deviation
Normalization:
    x = x - min/max-min
We will use normalization - it is recommended when there is a sigmoid in output

"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a datastructure with 60 time steps and 1 output
#As in the data one month has 20 days, and we will be looking at 60 time steps i.e. 3 months of historical data
X_train = []
y_train = []
#1258 - no. of training examples in the training_set
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping - to make X_train in the input format of keras recurrent layer
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM layer and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#units - number of lstm units(neurons) in the layer
#return_sequences - When we have multiple LSTM layers, we will set this to true for every layer except the last layer
#input_shape - the shape of the input

#Adding dropout:
regressor.add(Dropout(0.2))

#Stacking more LSTM layers:
#Only the first layer requires input_shape parameter.
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#We set return_sequences = False for the last layer(output layer), it's the default value.
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer:
regressor.add(Dense(units = 1))

#Compiling the RNN:
#Keras recommends rmsprop for RNNs, but in this case adam performs better
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set:
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Saving the model:
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#Serialize weights to HDF5:
regressor.save_weights("model.h5")

#Part 3 - Predictions
#Getting actual prices from the test set:
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock prices:
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

#reshaping for making RNN input compatible:
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the final results:
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
