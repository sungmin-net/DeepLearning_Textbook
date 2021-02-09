from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy
import os
import matplotlib.pyplot as plt
import tensorflow as tf

numpy.random.seed(3)
tf.random.set_seed(3)

data_pre = pd.read_csv('dataset/wine.csv', header=None)
data = data_pre.sample(frac = 0.15)

dataset = data.values;
x = dataset[:, 0:12]
y = dataset[:, 12]

# 모델의 설정
model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 저장 폴더 설정
dir = './192p_wineCheckpointGraph_models/'
if not os.path.exists(dir) :
    os.mkdir(dir)

# 모델 저장 조건 설정
modelpath = dir + "{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose = 1,
        save_best_only = True)

# 모델 실행 및 저장
history = model.fit(x, y, validation_split = 0.33, epochs = 3500, batch_size = 500,
        callbacks = [checkpointer]) # 책에 없는 callbacks 추가함

# y_vloss 에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss = history.history['val_loss']

# y_acc 에 학습셋으로 측정한 정확도의 값을 저장
y_acc = history.history['accuracy'] # 책에 있는 'acc' 를 'accuracy' 로 수정함

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = numpy.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c = "red", markersize = 3)
plt.plot(x_len, y_acc, "o", c = "blue", markersize = 3)
plt.show()