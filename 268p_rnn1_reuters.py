'''
201116_이 코드는 대체 무슨 의미인가.. 뉴스 데이터를 가져와서, 전처리를 하고, 
몇 개의 카테고리가 있는 지를 보고, 테스트셋과 학습셋을 나눈 다음에, 단어들을 보고
카테고리를 맞추도록 학습시키고, 테스트셋으로 카테고리를 잘 맞췄는 지 본다...? 
'''

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import matplotlib.pyplot as plt

# 불러온 데이터를 학습셋과 테스트셋으로 나누기 
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000, test_split = 0.2)

# 데이터 확인하기
category = numpy.max(y_train) + 1
print(category, ' 카테고리')
print(len(x_train), '학습용 뉴스 기사')
print(len(y_train), '테스트용 뉴스 기사')
print(x_train[0])

# 데이터 전처리
x_train = sequence.pad_sequences(x_train, maxlen = 100) # x 데이터(기사)에서 100개 단어만 가져옴
x_test = sequence.pad_sequences(x_test, maxlen = 100) # 100 보다 적으면 0으로 채움
y_train = np_utils.to_categorical(y_train) # y 데이터는 원핫인코딩
y_test = np_utils.to_categorical(y_test)

print(x_train)

# 모델 설정
model = Sequential()
# 임베딩 층은 입력된 값을 받아 다음층이 알아듣는 상태로 변환
# Embedding('불러온 단어의 총 개수', '기사당 단어 수')
model.add(Embedding(1000, 100))

# LSTM은 RNN 에서 기억 값에 대한 가중치를 제어
# LSTM('기사당 단어 수', 기타 옵션)
model.add(LSTM(100, activation = 'tanh')) 
                                   
model.add(Dense(46, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 실행
history = model.fit(x_train, y_train, batch_size = 100, epochs = 20,
        validation_data = (x_test, y_test))

# 테스트 정확도 출력
print('\n test accuracy : %.4f' % (model.evaluate(x_test, y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
