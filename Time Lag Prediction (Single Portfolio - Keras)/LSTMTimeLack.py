import pandas as pd
import numpy as np
from keras.optimizers import Nadam, RMSprop
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler
import csv
from numpy import linalg as LA

import h5json


# Isolating most representative portfolios via autoencoder
# then calibrating based only on those isolated portofolios
tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")

np.random.seed(7)

#Full = 648
periodNumber = 648
hiddenLayerNeuronNum = 300
epochs=400
numDataPoints = 100
split = 50
lookBackPeriod = 50
lookForwardPeriod = 8

exampleNum=7
timeWindow = 11

#using 0.25 as validation_split gives pretty good results
validation_split = 0.25

scaler = MinMaxScaler(feature_range=(0,1))
tenByTenData = scaler.fit_transform(tenByTenData)

def createLSTMTimeLack(dataSet, rownum):
    dataX = []
    dataY = []

    tempDataX = []
    dArray = dataSet[rownum]
    for i in range(len(dArray)-exampleNum-timeWindow-lookForwardPeriod+2):
        for j in range(0,exampleNum):
            temp=dArray[i+j:i+j+timeWindow]
            tempDataX.append(temp)
        newTempData = np.array(tempDataX)
        dataX.append(newTempData)
        tempDataX = []
    for i in range(len(dArray)-lookForwardPeriod-exampleNum-timeWindow+2):
        tempY = dArray[i+exampleNum+timeWindow-1:i+timeWindow+exampleNum+lookForwardPeriod-1]
        dataY.append(tempY)
    return dataX, dataY

x,y = createLSTMTimeLack(tenByTenData, 0)


trueX = np.asarray(x)
trueY = np.asarray(y)

testSplit = trueX.shape[0]*4/5
xTrain = trueX[:testSplit]
yTrain = trueY[:testSplit]
xTest = trueX[testSplit:]
yTest = trueY[testSplit:]

forecaster = Sequential()

forecaster.add(LSTM(timeWindow/2*3, input_shape=(exampleNum, timeWindow), return_sequences=False))
forecaster.add(Dense(lookForwardPeriod))
forecaster.compile(optimizer=RMSprop(lr=0.002), loss='mse', metrics=['accuracy'])

hist = forecaster.fit(xTrain,yTrain,epochs=100, shuffle=True)

# epoch_loss_forecaster = hist.history['loss']
# epoch_accuracy_forecaster = [x*100 for x in hist.history['acc']]
# epoch_num = [x for x in range(0,len(epoch_loss_forecaster))]
#
# plt.plot(epoch_num,epoch_loss_forecaster)
# plt.plot(epoch_num, epoch_accuracy_forecaster)
# plt.xlabel("Generation Number")
# plt.ylabel("Loss (Mean Squared Error)")
# plt.title("Forecaster Loss")
# plt.show()

l2Norms = []

for i in range(len(xTest)):
    l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTest[i:i+1])
    l2Norms.append(l2NormRatio)
#
# with open('timeLagTestResults.csv', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(l2Norms)

print np.sum(l2Norms)/len(l2Norms)

with open('quickDel.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(l2Norms)