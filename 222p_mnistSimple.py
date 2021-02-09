from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
# tf.set_random_seed(3)
tf.random.set_seed(3)

# MNIST 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 차원변경(2>1), 형변환, 정규화
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32') / 255 
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32') / 255

# 원핫 인코딩(바이너리화)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 모델 프레임 설정
model = Sequential()
model.add(Dense(512, input_dim = 28 * 28, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# 모델 실행환경 설정
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# 모델 최적화 설정
modelDir = './222p_mnistSimple_models/'
if not os.path.exists(modelDir) :
    os.mkdir(modelDir)
    
modelPath = modelDir + "{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelPath, monitor = 'val_loss', verbose = 1,
        save_best_only = True)
early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)

# 모델 실행
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30, 
        batch_size = 200, verbose = 0, callbacks = [checkpointer, early_stopper])

# 테스트 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = ".", c = "red", label = "Testset_loss")
plt.plot(x_len, y_loss, marker = ".", c = "blue", label = "Trainset_loss")
plt.legend(loc = "upper right")
#plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()