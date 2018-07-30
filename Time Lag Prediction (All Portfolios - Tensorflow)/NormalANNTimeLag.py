from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf

tenByTenData = pd.read_csv("100_Portfolios_ME_INV_10x10.CSV", index_col="Portfolio Names")

#Full = 648
periodNumber = 648
hiddenLayerNeuronNum = 300
numDataPoints = 100
split = 50
lookForwardPeriod = 8
batch_size = 4
epochs = 630

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

splitFactor = len(inputs)*4/5
xTrain = np.reshape(np.swapaxes(inputs[:splitFactor],1,2),newshape=(-1, numDataPoints, exampleNum*timeWindow))
yTrain = outputs[:splitFactor]
xTest = np.reshape(np.swapaxes(inputs[splitFactor:],1,2),newshape=(-1, numDataPoints, exampleNum*timeWindow))
yTest = outputs[splitFactor:]

x = tf.placeholder(tf.float32, shape=(batch_size,numDataPoints,exampleNum*timeWindow), name="inputs")
y_ = tf.placeholder(tf.float32, shape=(batch_size,numDataPoints,lookForwardPeriod), name="outputs")

W1 = tf.Variable(tf.truncated_normal(shape=[batch_size,exampleNum*timeWindow,(lookForwardPeriod+exampleNum*timeWindow)/2],stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape=[(lookForwardPeriod+exampleNum*timeWindow)/2]))

H1 = tf.nn.relu(tf.matmul(x, W1) + B1)

W2 = tf.Variable(tf.truncated_normal(shape=[batch_size,(lookForwardPeriod+exampleNum*timeWindow)/2,(lookForwardPeriod+exampleNum*timeWindow)/3],stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, shape=[(lookForwardPeriod+exampleNum*timeWindow)/3]))

H2 = tf.nn.relu(tf.matmul(H1, W2) + B2)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(H2, keep_prob)

W3 = tf.Variable(tf.truncated_normal(shape=[batch_size,(lookForwardPeriod+exampleNum*timeWindow)/3,lookForwardPeriod],stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, shape=[lookForwardPeriod]))

y = tf.matmul(h_fc1_drop, W3) + B3

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
    for epoch in range(epochs):
        batchMSEs = []
        batchL2s = []
        for batch_num in range(numTrainBatches):
            train_step.run(feed_dict={x: xTrain[batch_num], y_: yTrain[batch_num], keep_prob:0.5})
            batchMSEs.append(mse.eval(feed_dict={x: xTrain[batch_num], y_: yTrain[batch_num], keep_prob:0.5}))
            batchL2s.append(l2NormRatio.eval(feed_dict={x: xTrain[batch_num], y_:yTrain[batch_num], keep_prob:0.5}))
        print "Epoch", epoch, "MSE:", np.average(batchMSEs), "L2 Norm Ratio: ", np.average(batchL2s)

    print "Getting Testing Results"
    batchMSEs = []
    batchL2s = []
    for batch_num in range(numTestBatches):
        batchMSEs.append(mse.eval(feed_dict={x: xTest[batch_num], y_: yTest[batch_num], keep_prob:1.0}))
        batchL2s.append(l2NormRatio.eval(feed_dict={x: xTest[batch_num], y_: yTest[batch_num], keep_prob:1.0}))
    print "Testing MSE:", np.average(batchMSEs), "Testing L2 Norm Ratio: ", np.average(batchL2s)