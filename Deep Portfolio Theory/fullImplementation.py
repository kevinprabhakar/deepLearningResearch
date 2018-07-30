import pandas as pd
import numpy as np
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Nadam
from keras.layers import Input, Dense

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")

#Default = 648
startPeriod = 99
periodNumber = 648
hiddenLayerNeuronNum = 100
epochs=200
numDataPoints = 100
split = 55

#using 0.25 as validation_split gives pretty good results
validation_split = 0.25

data = tenByTenData.as_matrix()[:,:periodNumber]
labels = tenByTenData.as_matrix()[:,:periodNumber]

inputs = Input(shape=(periodNumber,))
encoder = Dense(hiddenLayerNeuronNum, activation='relu')(inputs)
predictions = Dense(periodNumber, activation='relu')(encoder)

autoencoder = Model(inputs=inputs, outputs=predictions)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

autoencoder.fit(data, labels, epochs=epochs, shuffle=True)

print "From " + tenByTenData.columns.values[startPeriod] + " to " + tenByTenData.columns.values[periodNumber]
print ""

print autoencoder.metrics_names

portfolioList = {}
sortedPortfolios = []

for i in range(numDataPoints):
    portfolioName = tenByTenData.index.tolist()[i:i+1][0]
    reconstructionAccuracy = autoencoder.test_on_batch(data[i:i+1,:],data[i:i+1,:])[1]
    portfolioList[portfolioName] = float(reconstructionAccuracy)
for key, value in sorted(portfolioList.iteritems(), key=lambda (k,v): (v,k)):
    sortedPortfolios.append(key)

tossOutPortfolios = sortedPortfolios[10:len(sortedPortfolios)-split]

nasdaqCompositePrices = pd.read_csv("NASDAQPrices.csv", parse_dates=[0], index_col="Date")

#stocks from this dataset come from NYSE, AMEX, and NASDAQ
tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

tenByTenDataByDate.drop(tossOutPortfolios, axis=1)

print tenByTenDataByDate.columns

inputNeurons = 100
hidden1Neurons = 80
hidden2Neurons = 50
hidden3Neurons = 20
outputNeurons=1

epochs2=500

activation = 'relu'

data = tenByTenDataByDate.as_matrix()[99:649,:]
labels = nasdaqCompositePrices.as_matrix()[:550,4:5]

inputs = Input(shape=(inputNeurons,))
hidden1 = Dense(hidden1Neurons, activation=activation)(inputs)
hidden2 = Dense(hidden2Neurons, activation=activation)(hidden1)
hidden3 = Dense(hidden3Neurons, activation=activation)(hidden2)
prediction = Dense(outputNeurons, activation=activation)(hidden3)

calibrator = Model(inputs, prediction)

opt = Nadam(lr=0.002)

tboard =TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)

calibrator.compile(optimizer=opt,loss='mae')

calibrator.fit(data,labels,epochs=epochs2, shuffle=True,callbacks=[tboard], validation_split=0.25)

evaluateFrom = 300
evaluateTo = 310

x = calibrator.predict_on_batch(data[evaluateFrom:evaluateTo])
margins = x-labels[evaluateFrom:evaluateTo]

errorPercentages = []

for i in range(len(margins)):
    error = (margins[i]/labels[i+evaluateFrom])*100
    errorPercentages.append(float(error))

print "Predictions"
print np.asarray(a=x,order=0)

print "Error Percentages"
print np.asarray(errorPercentages)

print "Average Absolute Error Percentage"
print float(np.sum(np.fabs(errorPercentages))/len(errorPercentages)), "%"

print "Margins"
print np.asarray(margins)

#calibrator 100 to hidden to 1
