from keras.datasets import mnist
from keras.utils import np_utils

import numpy
import sys
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)   # '이건 뭐........'

# mnist 데이터셋 불러오기
(x_train, y_train_class), (x_test, y_test_class) = mnist.load_data()

print("학습셋 이미지 수 : %d 개" % (x_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (x_test.shape[0]))

# # 그래프로 이미지 확인
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], cmap='Greys')
# plt.show()

# 코드로 확인
for x in x_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 차원 변환과정
# x_train = x_train.reshape(x_train.shape[0], 784) # 이미지 픽셀 사이즈, 28 x 28
x_train = x_train.reshape(x_train.shape[0], 28 * 28) # 이미지 픽셀 사이즈, 28 x 28
x_train = x_train.astype('float64') # 정규화 전 타입 변경
x_train = x_train / 255 # 정규화

# 테스트 값들에도 위 세줄과 동일한 적용
x_test = x_test.reshape(x_test.shape[0], 784).astype('float64') / 255

# 클래스 값 확인
print("class : %d" % (y_train_class[0]))

# 클래스 값들의 원핫 인코딩(바이너리화)
y_train = np_utils.to_categorical(y_train_class, 10)
y_test = np_utils.to_categorical(y_test_class, 10)

print(y_train[0])