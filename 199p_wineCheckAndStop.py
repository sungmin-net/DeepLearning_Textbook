from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy
import os
import tensorflow as tf

# seed 값 설정
numpy.random.seed(3)
tf.random.set_seed(3)

data_pre = pd.read_csv('dataset/wine.csv', header = None)
data = data_pre.sample(frac = 0.15)
dataset = data.values
x = dataset[:, 0:12]
y = dataset[:, 12]

model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 저장 폴더 만들기
modelDir = './199p_wineCheckAndStop_models/'
if not os.path.exists(modelDir):
    os.mkdir(modelDir)
    
modelpath = modelDir + "{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', verbose = 1,
        save_best_only = True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 100)

model.fit(x, y, validation_split = 0.2, epochs = 3500, batch_size = 500, verbose = 0,
        callbacks = [ early_stopping_callback, checkpointer]) 