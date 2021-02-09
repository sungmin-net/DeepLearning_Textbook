'''
실제가격과 관계없이 예상 가격이 모두 같다. jupyter notebook 도 마찬가지.
...
실제 가격 : 44.800, 예상 가격 : 6.336
실제 가격 : 17.100, 예상 가격 : 6.336
실제 가격 : 17.800, 예상 가격 : 6.336
..
결과가 엉망이다. 이걸 고칠 수 있어야 하는 데...TODO
epoch 를 늘리면 예상 가격이 실제 가격에 조금 더 근접해 지지만 여전히 모두 값이 같다.
input 인 x_test 는 모두 다른 값이긴 하다. 그런데 prediction 값이 모두 같다.
'''

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3) # '이게 무슨 또라이같은 코드인가..'

data = pd.read_csv("dataset/housing.csv", delim_whitespace = True, header = None)
# print(data.info())
# print(data.head())

'''
dataset = data.values
x = dataset[:, 0:13]
y = dataset[:, 13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)

model = Sequential()
model.add(Dense(30, input_dim = 13, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))

model.compile(loss = "mean_squared_error", optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 10)

# 예측 값과 실제 값의 비교
y_prediction = model.predict(x_test).flatten()
for i in range(10):
    label = y_test[i]
    prediction = y_prediction[i]
    print("실제 가격 : {:.3f}, 예상 가격 : {:.3f}".format(label, prediction))

print("print(x_test)------------------")
print(x_test)
print("print(model.predict(x_test))----------------------")
print(model.predict(x_test))
'''