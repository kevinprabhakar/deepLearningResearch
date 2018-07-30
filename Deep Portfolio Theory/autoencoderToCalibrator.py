import pandas as pd
import numpy as np
from keras.metrics import mae
from keras.optimizers import Nadam
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


from keras.layers import Input, Dense

tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

nasdaqCompositePrices = pd.read_csv("NASDAQPrices.csv", parse_dates=[0], index_col="Date")


Default = 648
numPortfolios = 100
hiddenLayerNeuronNum = 50
epochs=600
#
calibratorOutputNeurons = 1;
autoencoderActivation = 'relu'
calibratorActivation = 'relu'

data = tenByTenDataByDate.as_matrix()[:,:]

inputs1 = Input(shape=(numPortfolios,))
encoder1 = Dense(hiddenLayerNeuronNum, activation=autoencoderActivation)(inputs1)
predictions1 = Dense(numPortfolios, activation=autoencoderActivation)(encoder1)


autoencoder1 = Model(inputs=inputs1, outputs=predictions1)
autoencoder1.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
encoderHist = autoencoder1.fit(data, data, epochs=epochs, shuffle=True)

calibratorData = tenByTenDataByDate.as_matrix()[99:649,:]
calibratorOutputs = nasdaqCompositePrices.as_matrix()[:550,4:5]

calibInputs = Input(shape=(numPortfolios,))
calibHidden = Dense(hiddenLayerNeuronNum, activation=calibratorActivation)(calibInputs)
calibLabel = Dense(1, activation=calibratorActivation)(calibHidden)
autoCalibrator = Model(calibInputs,calibLabel)

opt = Nadam(lr=0.02)

autoCalibrator.compile(optimizer=opt, loss='mae')


history = autoCalibrator.fit(autoencoder1.predict_on_batch(calibratorData),calibratorOutputs, epochs=1000)


epoch_loss_calibrator = history.history['loss']
epoch_num = [x for x in range(0,len(epoch_loss_calibrator))]

encoderHistory = encoderHist.history['loss']
encoderEpochs = [x for x in range(0,len(encoderHistory))]


plt.figure(1)
plt.plot(epoch_num,epoch_loss_calibrator)
plt.xlabel("Generation Number")
plt.ylabel("Loss (Mean Absolute Error)")
plt.title("Calibrator Loss")

plt.figure(2)
plt.plot(encoderEpochs,encoderHistory)
plt.xlabel("Generation Number")
plt.ylabel("Loss (Mean Absolute Error)")
plt.title("Autoencoder Loss")
plt.show()



evaluateFrom = 300
evaluateTo = 310

x = autoCalibrator.predict_on_batch(calibratorData[evaluateFrom:evaluateTo])
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


