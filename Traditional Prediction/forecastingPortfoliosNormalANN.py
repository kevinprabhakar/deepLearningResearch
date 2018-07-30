import pandas as pd
import numpy as np
from keras.metrics import mae
from keras.optimizers import Nadam
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Input, Dense
import csv
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA

import math
from sklearn.metrics import mean_squared_error



# Isolating most representative portfolios via autoencoder
# then calibrating based only on those isolated portofolios
calibratorActivation = 'relu'

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")
tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")


#Full = 648
periodNumber = 648
hiddenLayerNeuronNum = 100
epochs=400
numDataPoints = 100
split = 45
lookBackPeriod = 50
lookForwardPeriod = 5


scaler = MinMaxScaler(feature_range=(0,1))
tenByTenData = scaler.fit_transform(tenByTenData)

def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(dataset.shape[1]-look_back-look_forward-1):
        dataX.append(dataset[:,i:i+look_back])
        dataY.append(dataset[:,i+look_back:i+look_back+look_forward])
    return np.array(dataX), np.array(dataY)

inputs, outputs = create_dataset(tenByTenData, lookBackPeriod, lookForwardPeriod)

print inputs.shape
print outputs.shape

testSplit = inputs.shape[0]*4/5
xTrain = inputs[:testSplit]
yTrain = outputs[:testSplit]
xTest = inputs[testSplit:]
yTest = outputs[testSplit:]

forecastInputs = Input(shape=(numDataPoints,lookBackPeriod))
forecastHidden = Dense(lookBackPeriod*3/5,activation='relu')(forecastInputs)
forecastHidden2 = Dense(lookBackPeriod*2/5,activation='relu')(forecastHidden)
forecastOutputs = Dense(lookForwardPeriod, activation='relu')(forecastHidden2)

forecaster = Model(forecastInputs,forecastOutputs)
forecaster.compile(optimizer=Nadam(lr=0.002), loss='mse', metrics=['accuracy'])

forecaster.fit(xTrain,yTrain,epochs=500, shuffle=True)

l2Norms = []

for i in range(len(xTest)):
    l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTest[i:i+1])
    l2Norms.append(l2NormRatio)

print np.average(l2Norms)
