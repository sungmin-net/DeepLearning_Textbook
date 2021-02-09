from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy
import os
import tensorflow as tf

# seed 값 설정
numpy.random.seed(3)
tf.random.set_seed(3)

data_pre = pd.read_csv('dataset/wine.csv', header=None)
data = data_pre.sample(frac = 1)

dataSet = data.values
x = dataSet[:, 0:12]
y = dataSet[:, 12]

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = './188p_wineCheckpoint_models/'
if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 1,
        save_best_only = True)

# 모델 실행 및 저장
model.fit(x, y, validation_split = 0.2, epochs = 200, batch_size = 200, verbose = 0,
        callbacks = [checkpointer])