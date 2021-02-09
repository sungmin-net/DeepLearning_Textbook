
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255 # 데이터정규화
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255 

# 생성자 모델
autoencoder = Sequential()

# 인코딩(입력된 차원을 축소)
autoencoder.add(Conv2D(16, kernel_size = 3, padding = 'same', input_shape = (28, 28, 1),
        activation = 'relu'))
autoencoder.add(MaxPooling2D(pool_size = 2, padding = 'same'))
autoencoder.add(Conv2D(8, kernel_size = 3, activation = 'relu', padding = 'same'))
autoencoder.add(MaxPooling2D(pool_size = 2, padding = 'same'))
autoencoder.add(Conv2D(8, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu'))

# 디코딩
autoencoder.add(Conv2D(8, kernel_size = 3, padding = 'same', activation = 'relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size = 3, padding = 'same', activation = 'relu'))
autoencoder.add(UpSampling2D())
# 아래 코드는 padding 이 없어서 벡터 값을 2만큼 줄임
autoencoder.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size = 3, padding = 'same', activation = 'sigmoid'))

# 전체 구조확인
autoencoder.summary()

# 컴파일 및 학습
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs = 5, batch_size = 128, validation_data = (x_test, x_test))
#autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 128, validation_data = (x_test, x_test))

# 학습된 결과 출력
random_test = np.random.randint(x_test.shape[0], size = 5) # 테스트할 이미지를 랜덤하게 불러옴
ae_imgs = autoencoder.predict(x_test) # 앞서 만든 오토인코더 모델에 입력

plt.figure(figsize = (7, 2)) # 출력될 이미지의 크기 설정

for i, image_idx in enumerate(random_test) : # 랜덤하게 뽑은 이미지를 차례대로 나열
    ax = plt.subplot(2, 7, i + 1)
    plt.imshow(x_test[ image_idx ].reshape(28, 28)) # 테스트할 이미지를 먼저 그대로 보여줌
    ax.axis('off')
    ax = plt.subplot(2, 7, 7 + i + 1)
    plt.imshow(ae_imgs[ image_idx ].reshape(28, 28)) # 오토인코딩 결과를 다음 열에 출력
    ax.axis('off')
plt.show()
