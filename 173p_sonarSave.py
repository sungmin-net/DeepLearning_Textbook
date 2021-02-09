# 이 코드도 안돌아가요..AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike'
# 이렇게 바뀜. ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

data = pd.read_csv('dataset/sonar.csv', header = None)

# print(data)
dataset = data.values
x = dataset[:, 0:60]
y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

# 학습셋과 테스트셋을 나눔
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs=130, batch_size=5)
model.save('my_model.h5') # 모델을 컴퓨터에 저장

del model # 테스트를 위해 메모리 내의 모델 삭제
model = load_model('my_model.h5') # 모델을 새로 불러옴

print("\n test accuracy : %.4f" % (model.evaluate(x_test, y_test)[1])) # 불러온 모델로 테스트 실행



