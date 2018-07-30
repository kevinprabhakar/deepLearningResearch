import pandas as pd
import numpy as np
from keras.optimizers import Nadam, RMSprop
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler
import csv
from numpy import linalg as LA

# Isolating most representative portfolios via autoencoder
# then calibrating based only on those isolated portofolios
tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")
tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

np.random.seed(7)

#Full = 648
periodNumber = 648
hiddenLayerNeuronNum = 300
epochs=400
numDataPoints = 100
split = 50
lookBackPeriod = 50
lookForwardPeriod = 5

scaler = MinMaxScaler(feature_range=(0,1))
tenByTenData = scaler.fit_transform(tenByTenData)

tenByTenData = tenByTenData[0:1,:periodNumber]
print tenByTenData

def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(dataset.shape[1]-look_back-look_forward-1):
        dataX.append(dataset[:,i:i+look_back])
        dataY.append(dataset[:,i+look_back:i+look_back+look_forward])
    return np.array(dataX), np.array(dataY)

data,labels = create_dataset(tenByTenData,lookBackPeriod,lookForwardPeriod)
testSplit = data.shape[0]*4/5
xTrain = data[:testSplit]
yTrain = labels[:testSplit]
xTest = data[testSplit:]
yTest = labels[testSplit:]
print data.shape
print labels.shape

# forecaster = Sequential()

forecastInputs = Input(shape=(xTrain.shape[1],lookBackPeriod))
forecastOutputs = Dense(lookForwardPeriod, activation='linear')(forecastInputs)
forecaster = Model(forecastInputs,forecastOutputs)

forecaster.compile(optimizer=Nadam(lr=0.002), loss='mse', metrics=['accuracy'])

forecaster.fit(xTrain,yTrain,epochs=100, shuffle=True)


l2Norms = []

for i in range(len(xTest)):
    l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTest[i:i+1])
    l2Norms.append(l2NormRatio)

print np.sum(l2Norms)/(float(len(l2Norms)))

# with open('singlePorfolioResults.csv', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(l2Norms)