# 이 코드도 안돌아감. AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike'
# 이렇게 바뀜. ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)
data = pd.read_csv('dataset/sonar.csv', header=None)
dataset = data.values
x = dataset[:, 0:60]
y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

# 학습셋과 테스트셋의 구분
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 130, batch_size = 5)

# 테스트셋 모델에 적용
print("\n Test accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))
