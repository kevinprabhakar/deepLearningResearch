import pandas as pd
import numpy as np
from keras.metrics import mae
from keras.optimizers import Nadam
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy.linalg as LA

import pprint


from keras.layers import Input, Dense


# Isolating most representative portfolios via autoencoder
# then calibrating based only on those isolated portofolios
calibratorActivation = 'relu'

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")
tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

nasdaqCompositePrices = pd.read_csv("NASDAQPrices.csv", parse_dates=[0], index_col="Date")
calibratorOutputNeurons = 1;

opt = Nadam(lr=0.02)

#Default = 648
startPeriod = 99
periodNumber = 648
hiddenLayerNeuronNum = 100
epochs=400
numDataPoints = 100
split = 45

#using 0.25 as validation_split gives pretty good results
validation_split = 0.25

data = tenByTenData.as_matrix()[:,:periodNumber]
labels = tenByTenData.as_matrix()[:,:periodNumber]

print data.shape
print labels.shape

inputs2 = Input(shape=(periodNumber,))
encoder2 = Dense(hiddenLayerNeuronNum, activation='relu')(inputs2)
predictions2 = Dense(periodNumber, activation='relu')(encoder2)

autoencoder2 = Model(inputs=inputs2, outputs=predictions2)
autoencoder2.compile(optimizer=Nadam(lr=0.002), loss='mse', metrics=['mean_squared_error'])

tboard =TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)
#
# hist = autoencoder2.fit(data[:400], labels[:400], epochs=epochs, shuffle=True, callbacks=[tboard])
#
# encoderHistory = hist.history['loss']
# encoderEpochs = [x for x in range(0,len(encoderHistory))]
#
# l2NormEncode = []
# l2NormEncodeTest = []
#
# for x in range(len(data[:300])):
#     i = x
#     l2NormRatio = LA.norm(autoencoder2.predict_on_batch(data[i:i+1])-labels[i:i+1])/LA.norm(labels[i:i+1])
#     l2NormEncode.append(l2NormRatio)
#
# for x in range(len(data[300:])):
#     i = x + 300
#     l2NormRatio = LA.norm(autoencoder2.predict_on_batch(data[i:i+1])-labels[i:i+1])/LA.norm(labels[i:i+1])
#     l2NormEncodeTest.append(l2NormRatio)
#
#
#
# print "From " + tenByTenData.columns.values[startPeriod] + " to " + tenByTenData.columns.values[periodNumber]
# print ""
#
# portfolioList = {}
# sortedPortfolios = []
#
# for i in range(numDataPoints):
#     portfolioName = tenByTenData.index.tolist()[i:i+1][0]
#
#     #test_on_batch returns scalar test loss
#     reconstructionAccuracy = autoencoder2.test_on_batch(data[i:i+1,:],data[i:i+1,:])[1]
#     portfolioList[portfolioName] = float(reconstructionAccuracy)
# for key, value in sorted(portfolioList.iteritems(), key=lambda (k,v): (v,k)):
#     sortedPortfolios.append(key)
#
# keepPortfolios = sortedPortfolios[0:10]+sortedPortfolios[len(sortedPortfolios)-split:]
#
# tenByTenDataByDate = tenByTenDataByDate[keepPortfolios]
#
# print len(tenByTenDataByDate.columns)
#
# calibratorData = tenByTenDataByDate.as_matrix()[99:649,:]
# calibratorOutputs = nasdaqCompositePrices.as_matrix()[:550,4:5]
#
# calibrateInputLayer = Input(shape=(len(tenByTenDataByDate.columns),))
# calibrateHiddenLayer = Dense(len(tenByTenDataByDate.columns)/2, activation=calibratorActivation)(calibrateInputLayer)
# calibrateLayer = Dense(calibratorOutputNeurons, activation=calibratorActivation)(calibrateHiddenLayer)
# calibrator = Model(inputs=calibrateInputLayer, outputs=calibrateLayer)
#
# calibratorEpochs = 1000
#
# tboard =TensorBoard(log_dir='./Graph', histogram_freq=0,
#           write_graph=True, write_images=True)
#
# calibrator.compile(optimizer=opt, loss='mae', metrics=[mae])
# hist = calibrator.fit(calibratorData[0:400], calibratorOutputs[0:400], epochs=calibratorEpochs, shuffle=True, callbacks=[tboard])
#
# epoch_loss_calibrator = hist.history['loss']
# epoch_num = [x for x in range(0,len(epoch_loss_calibrator))]
#
# plt.figure(1)
# plt.plot(epoch_num,epoch_loss_calibrator)
# plt.xlabel("Generation Number")
# plt.ylabel("Loss (Mean Absolute Error)")
# plt.title("Calibrator Loss")
#
# plt.figure(2)
# plt.plot(encoderEpochs,encoderHistory)
# plt.xlabel("Generation Number")
# plt.ylabel("Loss (Mean Absolute Error)")
# plt.title("Autoencoder Loss")
# plt.show()
#
#
#
# l2Norms = []
# l2NormsTesting = []
#
# calibratorSplit = 400
#
# for x in range(len(calibratorData[:400])):
#     i = x
#     l2NormRatio = LA.norm(calibrator.predict_on_batch(calibratorData[i:i+1])-calibratorOutputs[i:i+1])/LA.norm(calibratorOutputs[i:i+1])
#     l2Norms.append(l2NormRatio)
#
# for x in range(len(calibratorData[400:])):
#     i = x + 400
#     l2NormRatio = LA.norm(calibrator.predict_on_batch(calibratorData[i:i+1])-calibratorOutputs[i:i+1])/LA.norm(calibratorOutputs[i:i+1])
#     l2NormsTesting.append(l2NormRatio)
#
# print "Calibrator Testing Error:", (np.sum(l2NormsTesting)/len(l2NormsTesting))
# print "Calibrator Training Error:", (np.sum(l2Norms)/len(l2Norms))
# print "Autoencoder Testing Error:", (np.sum(l2NormEncodeTest)/len(l2NormEncodeTest))
# print "Autoencoder Training Error:", (np.sum(l2NormEncode)/len(l2NormEncode))

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(autoencoder2.to_json())

import json
import base64


a = np.zeros((3,3,3,3))
b = a.tolist()

print b
