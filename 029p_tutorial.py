import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 난수값 고정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 읽기
dataset = np.loadtxt("dataset\\ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X 와 Y 로 구분하여 저장
x = dataset[:,0:17]
y = dataset[:,17]

# 딥러닝 구조를 결정
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=10)
