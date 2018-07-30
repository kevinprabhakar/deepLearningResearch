import pandas as pd
import numpy as np
from keras.models import Model

from keras.layers import Input, Dense

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")


#Default = 648
startPeriod = 99
periodNumber = 648
hiddenLayerNeuronNum = 100
epochs=200
numDataPoints = 100
split = 15

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

sortedPortfolios = sortedPortfolios[:10]+sortedPortfolios[len(sortedPortfolios)-split:]

print sortedPortfolios