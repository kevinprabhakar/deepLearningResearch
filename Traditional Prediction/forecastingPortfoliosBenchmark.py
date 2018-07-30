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

#using 0.25 as validation_split gives pretty good results
validation_split = 0.25

scaler = MinMaxScaler(feature_range=(0,1))
tenByTenData = scaler.fit_transform(tenByTenData)

print tenByTenData

def create_dataset_LSTM(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(dataset.shape[1]-look_back-look_forward-1):
        dataX.append(dataset[:,i:i+look_back])
        dataY.append(dataset[:,i+look_back:i+look_back+look_forward])
    data, labels = np.array(dataX), np.array(dataY)
    return data, labels

data,labels = create_dataset_LSTM(tenByTenData,lookBackPeriod,lookForwardPeriod)
testSplit = data.shape[0]*4/5
xTrain = data[:testSplit]
yTrain = labels[:testSplit]
xTest = data[testSplit:]
yTest = labels[testSplit:]
print data.shape
print labels.shape

forecastInputs = Input(shape=(numDataPoints,lookBackPeriod))
forecastOutputs = Dense(lookForwardPeriod, activation='linear')(forecastInputs)
forecaster = Model(forecastInputs,forecastOutputs)
forecaster.compile(optimizer=RMSprop(lr=0.002), loss='mse', metrics=['accuracy'])

hist = forecaster.fit(xTrain,yTrain,epochs=500, shuffle=False)

epoch_loss_forecaster = hist.history['loss']
epoch_accuracy_forecaster = hist.history['acc']
epoch_num = [x for x in range(0,len(epoch_loss_forecaster))]

plt.plot(epoch_num,epoch_loss_forecaster)
plt.plot(epoch_num, epoch_accuracy_forecaster)
plt.xlabel("Generation Number")
plt.ylabel("Loss (Mean Squared Error)")
plt.title("Forecaster Loss")
plt.show()

l2Norms = []

for i in range(len(xTest)):
    l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTest[i:i+1])
    l2Norms.append(l2NormRatio)

print np.average(l2Norms)
#
# with open(r'outputToDeleteBenchMark.csv', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(l2Norms)