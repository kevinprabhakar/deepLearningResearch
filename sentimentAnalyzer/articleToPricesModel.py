from keras.layers import Input, Dense
from keras.models import Model

inputLayer = Input(shape=(300,))
dense1 = Dense(150, activation='relu',use_bias=True)(inputLayer)

model = Model(inputLayer, dense1)