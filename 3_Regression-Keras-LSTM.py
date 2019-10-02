import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import math
import datetime

timesteps = seq_length = 30
sequence_length = seq_length + 1
data_dim = 3
hidden_dim = 4
output_dim = 1

xy = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/1_dataset.csv",delimiter=',')

xy = xy[::-1]
x = xy
scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)
y = xy[:,[-1]]
x = np.array(x)
y = np.array(y)

xshape = []
for index in range(len(xy) - seq_length):
    xshape.append(x[index: index + seq_length])
xshape = np.array(xshape)

yshape = np.zeros([xshape.shape[0],4])
for index in range(len(yshape)):
    for col in range(4):
        yshape[index,col] = y[index+col*10]



y0s = np.array(yshape[:,0])
y0s = np.reshape(y0s,(y0s.shape[0],1))
y1s = np.array(yshape[:,1])
y1s = np.reshape(y1s,(y1s.shape[0],1))
y2s = np.array(yshape[:,2])
y2s = np.reshape(y2s,(y2s.shape[0],1))
y3s = np.array(yshape[:,3])
y3s = np.reshape(y3s,(y3s.shape[0],1))

yshape = y3s



print(xshape.shape, yshape.shape, y0s.shape, y1s.shape)

train_size = int(round(xshape.shape[0]*0.8))
trainX = xshape[:train_size,:]
trainX = np.reshape(trainX,(trainX.shape[0],trainX.shape[1],trainX.shape[2]))
trainY = yshape[:train_size]
testX = xshape[train_size:,:]
testX = np.reshape(testX,(testX.shape[0],testX.shape[1],testX.shape[2]))
testY = yshape[train_size:]


# xy1 = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/1_dataset-1.csv",delimiter=',')
# xy1 = xy1[::-1]
# x1 = xy1
# scaler1 = StandardScaler()
# scaler1.fit(x)
# x1 = scaler1.fit_transform(x1)
# y1 = xy1[:, [-1]]
# x1 = np.array(x1)
# y1 = np.array(y1)
#
# xshape1 = []
# for index in range(len(xy1) - seq_length):
#     xshape1.append(x1[index: index + seq_length])
# xshape1 = np.array(xshape1)
#
# yshape1 = np.zeros([xshape1.shape[0], 4])
# for index in range(len(yshape1)):
#     for col in range(4):
#         yshape1[index, col] = y1[index + col * 10]
#
# y0s1 = np.array(yshape1[:, 0])
# y0s1 = np.reshape(y0s1, (y0s1.shape[0], 1))
# y1s1 = np.array(yshape1[:, 1])
# y1s1 = np.reshape(y1s1, (y1s1.shape[0], 1))
# y2s1 = np.array(yshape1[:, 2])
# y2s1 = np.reshape(y2s1, (y2s1.shape[0], 1))
# y3s1 = np.array(yshape1[:, 3])
# y3s1 = np.reshape(y3s1, (y3s1.shape[0], 1))
# yshape1 = y3s1
#
# train_size1 = int(round(xshape1.shape[0]*0.8))
# testX1 = xshape1[train_size1:,:]
# testX1 = np.reshape(testX1,(testX1.shape[0],testX1.shape[1],testX1.shape[2]))
# testY1 = yshape1[train_size1:]
# np.savetxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_testY1.csv",testY1,delimiter=',')
# np.savetxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_testY.csv",testY,delimiter=',')
# # model.fit(trainX,trainY,validation_data=(testX1, testY1),batch_size=1,epochs=5)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

model = Sequential()
model.add(LSTM(30, return_sequences=True, input_shape=(30,4)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(trainX,trainY,validation_data=(testX, testY),batch_size=1,epochs=5)

pred = model.predict(testX)
np.savetxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch5.csv",pred,delimiter=',')


