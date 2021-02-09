# 잘 돌아가는 데 (GPU가 없어서 굉장히) 오래 걸림
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 차원변경(2>1), 형변환, 정규화
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 컨볼루션 신경망 설정
model = Sequential()
# 3x3 의 커널로 슬라이딩 윈도우를 적용, input_shape 는 (행, 열, 색상 - 흑백은 1, 색상은 3)
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# 2x2 의 커널마다 가장 큰 값을 추출하여 적용. pool_size 가 2면 크기가 절반으로 줄어듦
model.add(MaxPooling2D(pool_size = 2))
# 노드의 25% 를 끔
model.add(Dropout(0.25))
# Dense 는 1차원 배열로 바꿔주어야 활성화 함수를 사용할 수 있으므로, 차원 변경
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
# 10개의 출력을 위해 softmax 사용
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 최적화 설정
modelDir = "./235p_mnistCnn_models/"
if not os.path.exists(modelDir) :
    os.mkdir(modelDir)

modelPath = modelDir + "{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', verbose = 1,
        save_best_only = True)
earlyStopper = EarlyStopping(monitor = 'val_loss', patience = 10)

# 모델의 실행
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 30,
        batch_size = 200, verbose = 0, callbacks = [earlyStopper, checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = ".", c = "red", label = "Testset_loss")
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()