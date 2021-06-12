import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise


kdBP = [0., 16., 35.]
kdV = [0.05, 0.935, 1.65]

kpV = [1.2, 0.8, 0.5]
kiV = [0.18, 0.12]

x = np.array([[1.2, 0.8, 0.5] for _ in range(5000)])
y = np.array([[0.05, 0.935, 1.65] for _ in range(5000)])

model = Sequential()
model.add(GaussianNoise(0.03, input_shape=(3,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3))

model.compile('adam', loss='mae', metrics='mse')
model.fit(x, y, epochs=1000, batch_size=16)
