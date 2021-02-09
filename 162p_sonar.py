# 이 코드 안돌아감
'''
# 내가 친것.--------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
data = pd.read_csv('dataset/sonar.csv', header=None)
# print(data)

# # 데이터 개괄 보기
# print(data.info())


# # 데이터 일부분 보기
# print(data.head())

dataSet = data.values
# print(dataset)
x = dataSet[:, 0:60]
y_obj = dataSet[:, 60]

# print(x)
# print("---------------")
# print(y_obj)

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 실행
model.fit(x, y, epochs = 200, batch_size = 5)

# 결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(x, y)[1]))
'''

# 교재 코드 복사 ---------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv('dataset/sonar.csv', header=None) # 파일은 연결해줌

# 데이터 개괄 보기
print(df.info())

# 데이터의 일부분 미리 보기
print(df.head())

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행 - 안됨
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
model.fit(X, Y, epochs=200, batch_size=5)
 
# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))