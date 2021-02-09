from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
data = pd.read_csv('dataset/iris.csv', names=["sepal_length", "sepal_width", "petal_length",
        "petal_width", "species"])

# # 그래프로 확인
# sns.pairplot(data, hue='species');
# plt.show()

dataSet = data.values
x = dataSet[:, 0:4].astype(float) # 모든 라인의 0~4 까지(4개, 4는 뺌) 인덱스인가? -> 맞음 
#print(x)
# print("--------------")

y_obj = dataSet[:, 4]
# print(y_obj)

# print("--------------")

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)
# print(y)

y_encoded = tf.keras.utils.to_categorical(y) # 단순 list 가 아니라 별도의 type 같음
print(y_encoded)

# 모델 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(x, y_encoded, epochs=50, batch_size=1)

''' 173p_sonarSave.py 가 안돌아가서 해봄
model.save('my_model.h5') # 모델을 컴퓨터에 저장
del model
model = load_model('my_model.h5') # 모델을 새로 불러옴
'''

# 결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(x, y_encoded)[1]))
