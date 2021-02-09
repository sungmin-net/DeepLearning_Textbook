import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.read_csv("dataset/pima-indians-diabetes.csv", names=['pregnant', 'plasma', 'pressure',
#        'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])

#print(data.head(5))

# print('\n')
# print(data.info()) # 데이터의 전반적인 정보 확인

# print('\n') # 각 정보별 특징을 좀 더 자세히 출력
# print(data.describe())

# print("\n")
# print(data[['pregnant', 'class']]) # 데이터 중 임신 정보와 클래스만 출력

# print('\n')
# print(data[['pregnant', 'class',]].groupby(['pregnant'], as_index=False)
#         .mean().sort_values(by='pregnant', ascending=True))

# 데이터 간의 상관관계를 그래프로 표현
# colormap = plt.cm.gist_heat  # 그래프의 색상 구성 선택
# plt.figure(figsize=(12, 12)) # 그래프의 크기

# # 그래프의 속성 결정. vmax 값을 0.5로 지정해 0.5하면 0.5에 가까울 수록 붉은 색
# sns.heatmap(data.corr(), linewidth=0.1, vmax = 0.5, cmap = colormap, linecolor='white', annot=True)
# plt.show()

# grid = sns.FacetGrid(data, col='class')
# grid.map(plt.hist, 'plasma', bins=10)
# plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy
import tensorflow as tf

numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataSet = numpy.loadtxt("dataset/pima-indians-diabetes.csv", delimiter=",")
x = dataSet[:, 0:8] # 한줄에 9개, 0부터 7까지 8개.
y = dataSet[:, 8] # 마지막 클래스

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# 모델 실행
model.fit(x, y, epochs=200, batch_size=10)

# 결과 출력
print("\n accuracy : %.4f" % (model.evaluate(x, y)[1]))