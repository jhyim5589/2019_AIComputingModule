import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import math
import datetime

#데이터셋은 0.1초 단위로 취득되어 직전 3초간 30개의 데이터셋을 기반으로 하여 향후 3초 거리를 예측함
timesteps = seq_length = 30 #batch를 구성한 데이터셋 길이를 의미
sequence_length = seq_length + 1
data_dim = 3
hidden_dim = 4
output_dim = 1

#저장되어 있는 데이터셋을 numpy array로 불러옴
xy = np.loadtxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/1_dataset.csv",delimiter=',')

# x, y에 대하여 x는 현재거리를 포함한 xy전체로 설정하고 standard scaling(m=0,std=1) 수행
# y는 우선 xy의 마지막열(현재 거리)로 설정
x = xy
scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)
y = xy[:,[-1]]
x = np.array(x)
y = np.array(y)

# LSTM에 인가하기 위한 3차원 배열 batch 구성을 위하여 shaping수행
# 한편 y 또한 현재, 향후1초, 2초, 3초에 대하여 각각 shift된 배열로 표현
xshape = []
for index in range(len(xy) - seq_length):
    xshape.append(x[index: index + seq_length])
xshape = np.array(xshape)

yshape = np.zeros([xshape.shape[0],4])
for index in range(len(yshape)):
    for col in range(4):
        yshape[index,col] = y[index+col*10]


# train 및 test y로 사용될 컬럼을 아래와 같이  현재, 향후 1초, 2초, 3초의 경우에 따라 다른 변수에 저장
y0s = np.array(yshape[:,0])
y0s = np.reshape(y0s,(y0s.shape[0],1))
y1s = np.array(yshape[:,1])
y1s = np.reshape(y1s,(y1s.shape[0],1))
y2s = np.array(yshape[:,2])
y2s = np.reshape(y2s,(y2s.shape[0],1))
y3s = np.array(yshape[:,3])
y3s = np.reshape(y3s,(y3s.shape[0],1))

yshape = y3s # 향후 3초의 거리를 예측



print(xshape.shape, yshape.shape, y0s.shape, y1s.shape)

# 기존 데이터에 대하여 train 80%, test 20%로 구성
train_size = int(round(xshape.shape[0]*0.8))
trainX = xshape[:train_size,:]
trainX = np.reshape(trainX,(trainX.shape[0],trainX.shape[1],trainX.shape[2]))
trainY = yshape[:train_size]
testX = xshape[train_size:,:]
testX = np.reshape(testX,(testX.shape[0],testX.shape[1],testX.shape[2]))
testY = yshape[train_size:]


print(trainX.shape, trainY.shape, testX.shape, testY.shape)

Sequential model에 2 LSTM layer, 1 dense layer로 구성
model = Sequential()
model.add(LSTM(30, return_sequences=True, input_shape=(30,4)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

# 이미 3차원 array를 통하여 30개의 데이터셋에 대한 batch를 구성하였으므로 batch size는 1로, epoch은 5로 설정하여 fitting수행
model.fit(trainX,trainY,validation_data=(testX, testY),batch_size=1,epochs=5)

# prediction 결과를 별도로 저장
pred = model.predict(testX)
np.savetxt("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/4_epoch5.csv",pred,delimiter=',')


