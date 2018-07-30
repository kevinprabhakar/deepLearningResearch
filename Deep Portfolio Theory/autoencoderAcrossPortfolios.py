import pandas as pd
import numpy as np
from keras.metrics import mae
from keras.optimizers import Nadam
from keras.models import Model
from keras.callbacks import TensorBoard


from keras.layers import Input, Dense

tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

nasdaqCompositePrices = pd.read_csv("NASDAQPrices.csv", parse_dates=[0], index_col="Date")


#Default = 648
numPortfolios = 100
hiddenLayerNeuronNum = 50
epochs=200

calibratorOutputNeurons = 1;
autoencoderActivation = 'relu'
calibratorActivation = 'relu'

data = tenByTenDataByDate.as_matrix()[:,:]

inputs1 = Input(shape=(numPortfolios,))
encoder1 = Dense(hiddenLayerNeuronNum, activation=autoencoderActivation)(inputs1)
predictions1 = Dense(numPortfolios, activation=autoencoderActivation)(encoder1)

calibrateLayer = Dense(calibratorOutputNeurons, activation=calibratorActivation)(encoder1)

autoencoder1 = Model(inputs=inputs1, outputs=predictions1)
calibrator = Model(inputs=inputs1, outputs=calibrateLayer)
autoencoder1.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

autoencoder1.fit(data, data, epochs=epochs, shuffle=True)

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")

#Default = 648
startPeriod = 99
periodNumber = 648
hiddenLayerNeuronNum = 300
epochs=200
numDataPoints = 100
split = 15

#using 0.25 as validation_split gives pretty good results
validation_split = 0.25

data = tenByTenData.as_matrix()[:,:periodNumber]
labels = tenByTenData.as_matrix()[:,:periodNumber]

inputs2 = Input(shape=(periodNumber,))
encoder2 = Dense(hiddenLayerNeuronNum, activation='relu')(inputs2)
predictions2 = Dense(periodNumber, activation='relu')(encoder2)

autoencoder2 = Model(inputs=inputs2, outputs=predictions2)
autoencoder2.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

autoencoder2.fit(data, labels, epochs=epochs, shuffle=True)

print "From " + tenByTenData.columns.values[startPeriod] + " to " + tenByTenData.columns.values[periodNumber]
print ""

portfolioList = {}
sortedPortfolios = []

for i in range(numDataPoints):
    portfolioName = tenByTenData.index.tolist()[i:i+1][0]

    #test_on_batch returns scalar test loss
    reconstructionAccuracy = autoencoder2.test_on_batch(data[i:i+1,:],data[i:i+1,:])[1]
    portfolioList[portfolioName] = float(reconstructionAccuracy)
for key, value in sorted(portfolioList.iteritems(), key=lambda (k,v): (v,k)):
    sortedPortfolios.append(key)

keepPortfolios = sortedPortfolios[0:10]+sortedPortfolios[len(sortedPortfolios)-split:]

print "YO" , tenByTenDataByDate.shape


tenByTenDataByDate = tenByTenDataByDate[keepPortfolios]

print "YO2" , tenByTenDataByDate.shape



calibratorData = tenByTenDataByDate.as_matrix()[99:649,:]
calibratorOutputs = nasdaqCompositePrices.as_matrix()[:550,4:5]
calibratorEpochs = 1000
opt = Nadam(lr=0.002)

tboard =TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)

calibrator.compile(optimizer=opt, loss='mae', metrics=[mae])
calibrator.fit(calibratorData, calibratorOutputs, epochs=calibratorEpochs, shuffle=True, callbacks=[tboard])


evaluateFrom = 300
evaluateTo = 310

x = calibrator.predict_on_batch(calibratorData[evaluateFrom:evaluateTo])
margins = x-calibratorOutputs[evaluateFrom:evaluateTo]

errorPercentages = []

for i in range(len(margins)):
    error = (margins[i]/calibratorOutputs[i+evaluateFrom])*100
    errorPercentages.append(float(error))

print "Predictions"
print np.asarray(a=x,order=0)

print "Error Percentages"
print np.asarray(errorPercentages)

print "Average Absolute Error Percentage"
print float(np.sum(np.fabs(errorPercentages))/len(errorPercentages)), "%"

print "Margins"
print np.asarray(margins)