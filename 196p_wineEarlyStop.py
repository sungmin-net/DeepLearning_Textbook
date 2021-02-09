from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy
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

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 100)

# 모델 실행
model.fit(x, y, validation_split = 0.2, epochs = 2000, batch_size = 500,
        callbacks=[early_stopping_callback])

# 결과 출력
print("\n accuracy : %.4f" % (model.evaluate(x, y)[1])) # 이게 무슨 의미인가...