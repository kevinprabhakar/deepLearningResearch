from keras.models import Model
from keras import metrics
from keras.callbacks import TensorBoard
from keras.optimizers import Nadam
from keras.layers import Input, Dense
import numpy as np
import pandas as pd

nasdaqCompositePrices = pd.read_csv("NASDAQPrices.csv", parse_dates=[0], index_col="Date")

#stocks from this dataset come from NYSE, AMEX, and NASDAQ
tenByTenDataByDate = pd.read_csv("100_Portfolios_ME_INV_10x10_byDate.CSV", index_col="Portfolio Names")

inputNeurons = 100
hidden1Neurons = 80
hidden2Neurons = 50
hidden3Neurons = 20
outputNeurons=1

epochs=500

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

calibrator.compile(optimizer=opt,loss='mae',metrics=[metrics.mae])

calibrator.fit(data,labels,epochs=epochs, batch_size=20, shuffle=True,callbacks=[tboard])


x = calibrator.predict_on_batch(data[300:320])
print x
# print data[:10]
print x-labels[300:320]
margins = x-labels[300:320]

errorPercentages = []

for i in range(len(margins)):
    error = (margins[i]/labels[i+300])*100
    errorPercentages.append(float(error))

print errorPercentages

