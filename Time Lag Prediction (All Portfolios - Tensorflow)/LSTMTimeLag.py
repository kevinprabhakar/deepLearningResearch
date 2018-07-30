from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")

#Full = 648
periodNumber = 648
hiddenLayerNeuronNum = 300
numDataPoints = 100
split = 50
lookForwardPeriod = 8
batch_size = 4
epochs = 600

exampleNum=7
timeWindow = 11

scaler = MinMaxScaler(feature_range=(0,1))
tenByTenData = scaler.fit_transform(tenByTenData)

def createTimeLag(dataSet):
    dataX = []
    dataY = []

    tempDataX = []
    for i in range(len(dataSet)-exampleNum-timeWindow-lookForwardPeriod+2):
        for j in range(0,exampleNum):
            temp=dataSet[:,i+j:i+j+timeWindow]
            tempDataX.append(temp)
        newTempData = np.array(tempDataX)
        dataX.append(newTempData)
        tempDataX = []
    for i in range(len(dataSet)-lookForwardPeriod-exampleNum-timeWindow+2):
        tempY = dataSet[:,i+exampleNum+timeWindow-1:i+timeWindow+exampleNum+lookForwardPeriod-1]
        dataY.append(tempY)
    return np.asarray(dataX), np.asarray(dataY)

inputs, outputs = createTimeLag(tenByTenData)

splitFactor = len(inputs)*7/10
xTrain = np.reshape(inputs[:splitFactor],newshape=(-1, numDataPoints, exampleNum*timeWindow))
yTrain = outputs[:splitFactor]
xTest = np.reshape(inputs[splitFactor:],newshape=(-1, numDataPoints, exampleNum*timeWindow))
yTest = outputs[splitFactor:]

print "Input Training Data Shape: ", xTrain.shape
print "Output Training Data Shape: ", yTrain.shape
print "Input Testing Data Shape: ", xTest.shape
print "Output Testing Data Shape: ", yTest.shape

x = tf.placeholder(tf.float32, shape=[None, numDataPoints, exampleNum*timeWindow], name="inputs")
y_ = tf.placeholder(tf.float32, shape=[None, numDataPoints, lookForwardPeriod], name="outputs")

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=8, activation=tf.nn.relu)

y, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

mse = tf.reduce_mean(tf.squared_difference(y, y_))
l2NormRatio = tf.reduce_mean(tf.norm(y-y_)/tf.norm(y_))

train_step = tf.train.AdamOptimizer().minimize(mse)

init = tf.global_variables_initializer()

numTrainBatches = len(xTrain)/batch_size
xTrain = np.array_split(xTrain, numTrainBatches)
yTrain = np.array_split(yTrain, numTrainBatches)

numTestBatches = len(xTest)/batch_size
xTest = np.array_split(xTest, numTestBatches)
yTest = np.array_split(yTest, numTestBatches)

with tf.Session() as sess:
    init.run()
    trainingBatchMSEs = []
    trainingBatchL2s = []
    for epoch in range(epochs):
        batchMSEs = []
        batchL2s = []
        for batch_num in range(numTrainBatches):
            train_step.run(feed_dict={x: xTrain[batch_num], y_: yTrain[batch_num]})
            batchMSEs.append(mse.eval(feed_dict={x: xTrain[batch_num], y_: yTrain[batch_num]}))
            batchL2s.append(l2NormRatio.eval(feed_dict={x: xTrain[batch_num], y_:yTrain[batch_num]}))
        print "Epoch", epoch, "MSE:", np.average(batchMSEs), "L2 Norm Ratio: ", np.average(batchL2s)
        trainingBatchMSEs.append(np.average(batchMSEs))
        trainingBatchL2s.append(np.average(batchL2s))

    print "\nGetting Testing Results"
    batchMSEs = []
    batchL2s = []
    for batch_num in range(numTestBatches):
        batchMSEs.append(mse.eval(feed_dict={x: xTest[batch_num], y_: yTest[batch_num]}))
        batchL2s.append(l2NormRatio.eval(feed_dict={x: xTest[batch_num], y_: yTest[batch_num]}))
    print "Testing MSE:", np.average(batchMSEs), "Testing L2 Norm Ratio: ", np.average(batchL2s)



    plt.subplot(2, 1, 1)
    plt.plot(range(epochs),trainingBatchL2s)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss (L2 Norm Ratio of Error)")

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(batchL2s)), batchL2s)
    plt.xlabel("Grouped Batch Number")
    plt.ylabel("Loss (L2 Norm Ratio of Error)")

    plt.show()