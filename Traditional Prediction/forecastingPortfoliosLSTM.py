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

data = tenByTenData.as_matrix()[:,:periodNumber]
labels = tenByTenData.as_matrix()[:,:periodNumber]

inputs2 = Input(shape=(periodNumber,))
encoder2 = Dense(hiddenLayerNeuronNum, activation='relu')(inputs2)
predictions2 = Dense(periodNumber, activation='relu')(encoder2)

autoencoder2 = Model(inputs=inputs2, outputs=predictions2)
autoencoder2.compile(optimizer=Nadam(lr=0.002), loss='mse', metrics=['accuracy'])

hist = autoencoder2.fit(data, labels, epochs=epochs, shuffle=True)

encoderHistory = hist.history['loss']
encoderEpochs = [x for x in range(0,len(encoderHistory))]

portfolioList = {}
sortedPortfolios = []

for i in range(numDataPoints):
    portfolioName = tenByTenData.index.tolist()[i:i+1][0]

    #test_on_batch returns scalar test loss
    reconstructionAccuracy = LA.norm(autoencoder2.predict_on_batch(data[i:i+1])-labels[i:i+1])/LA.norm(labels[i:i+1])
    portfolioList[portfolioName] = float(reconstructionAccuracy)
for key, value in sorted(portfolioList.iteritems(), key=lambda (k,v): (v,k)):
    sortedPortfolios.append(key)

keepPortfolios = sortedPortfolios[0:10]+sortedPortfolios[len(sortedPortfolios)-split:]
tossPortfolios = sortedPortfolios[10:len(sortedPortfolios)-split]

tenByTenData = tenByTenData.drop(tossPortfolios)

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

forecaster = Sequential()

forecaster.add(LSTM(lookBackPeriod/2*3, input_shape=(10+split, lookBackPeriod), return_sequences=True))
forecaster.add(TimeDistributed(Dense(lookForwardPeriod),input_shape=(lookBackPeriod/2,)))
forecaster.compile(optimizer=RMSprop(lr=0.002), loss='mse', metrics=['accuracy'])

hist = forecaster.fit(xTrain,yTrain,epochs=100, shuffle=False)

epoch_loss_forecaster = hist.history['loss']
epoch_accuracy_forecaster = hist.history['acc']
epoch_num = [x for x in range(0,len(epoch_loss_forecaster))]

plt.plot(epoch_num,epoch_loss_forecaster)
plt.plot(epoch_num, epoch_accuracy_forecaster)
plt.xlabel("Generation Number")
plt.ylabel("Loss (Mean Squared Error)")
plt.title("Forecaster Loss")
plt.show()

# l2Norms = []
#
# for i in range(len(xTest)):
#     l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTest[i:i+1])
#     l2Norms.append(l2NormRatio)
#
# epoch_num = [x for x in range(0,len(l2Norms))]
#
# plt.plot(epoch_num, l2Norms)
# plt.xlabel("Generation Number")
# plt.ylabel("Loss (Mean Squared Error)")
# plt.title("Forecaster Loss")
# plt.show()


l2Norms = []

for i in range(len(xTest)):
    l2NormRatio = LA.norm(forecaster.predict_on_batch(xTest[i:i+1])-yTest[i:i+1])/LA.norm(yTrain[i:i+1])
    l2Norms.append(l2NormRatio)

print np.average(l2Norms)
