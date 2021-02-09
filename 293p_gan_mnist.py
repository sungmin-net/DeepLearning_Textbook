from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

import numpy as np
import matplotlib.pyplot as plt

# 이미지 저장 경로가 없으면 생성
import os
if not os.path.exists('./293p_gan_mnist'):
    os.makedirs('./293p_gan_mnist')
    
# 생성자 모델
generator = Sequential()
# 128은 임의로 정한 노드 수. input_dim = 100 은 임의로 100 차원의 랜덤 백터를 준비하라는 뜻.
# 7 * 7 은 이후 UpSampling2D 를 두 번 거치면서 28 * 28 의 크기가 된다는 뜻. 
generator.add(Dense(128 * 7 * 7, input_dim = 100, activation = LeakyReLU(0.2)))
generator.add(BatchNormalization()) # 데이터의 배치를 정규 분포로 만든다고 함
generator.add(Reshape((7, 7, 128))) # Conv2D 가 받아들일 수 있는 형태로 바꿔주는 코드
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size = 5, padding = 'same')) # 커널은 5 x 5 크기(왜 책 284p은 3인가..)
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size = 5, padding = 'same', activation = 'tanh')) # 판별자로 넘길 준비

# 판별자 모델
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size = 5, strides = 2, input_shape = (28, 28, 1), 
        padding = 'same')) # strides = 2 는 커널을 두 칸씩 이동 
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size = 5, strides = 2, padding = 'same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten()) #  2차원을 1차원으로 변경
discriminator.add(Dense(1, activation = 'sigmoid')) # 활성화 함수
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam') # 로스 함수와 최적화 함수
discriminator.trainable = False # 판별자 자신이 학습되는 것을 막음

# 생성자와 판별자 모델을 연결시키는 gan 모델
gen_input = Input(shape = (100,)) # 임의의 100개 벡터 준비
dis_output = discriminator(generator(gen_input)) # 생성자 모델을 판별자에 입력
gan = Model(gen_input, dis_output)
# 참/거짓을 구별하는 이진 로스 함수와 최적화 함수를 넣고 컴파일
gan.compile(loss = 'binary_crossentropy', optimizer = 'adam') 
gan.summary()

# 신경망을 실행시키는 함수
def gan_train(epoch, batch_size, saving_interval):
    # MNIST 데이터 불러오기(이미지만 쓸 것이므로, x_train만 사용)
    (x_train, _), (_, _) = mnist.load_data()
    # 가로 28, 세로 28 픽셀이고, 흑백이므로 1을 설정
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    # 픽셀값(0 ~ 255)을 -1 ~ 1 사이로 정규화(분자에서 중앙값을 빼고, 중앙 값으로 나눔)
    # (0 ~ 1 정규화(분자에서 최소값을 빼고, 최대값으로 나눔)의 변형)
    x_train = (x_train - 127.5) / 127.5
    
    trues = np.ones((batch_size, 1))
    falses = np.zeros((batch_size, 1))
    
    for i in range(epoch) :
        # 실제 데이터를 판별자에 입력.
        # randint(a, b, c) 는 a 부터 b 까지의 숫자 중 하나를 랜덤하게 c 번 반복해서 가져오라는 뜻
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[ idx ]  # 실제 이미지를 랜덤하게 선택
        # 모두 참(1) 이라는 레이블을 붙이고, 딱 한번만 학습
        d_loss_real = discriminator.train_on_batch(imgs, trues)  
                
        # 가상 이미지를 판별자에 입력
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        # 모두 가짜라는 레이블을 붙임
        d_loss_fake = discriminator.train_on_batch(gen_imgs, falses)
        
        # 판별자와 생성자의 오차를 계산(d_loss_real 과 d_loss_fake 의 값을 모두 더해 둘로 나눔)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # 292p 에는 gen_imgs 를 쓴다고 하는 데 왜 코드는 noise 를 쓰는 가? 무엇이 맞는 가??
        g_loss = gan.train_on_batch(noise, trues)
        
        print('epoch : %d' % i, ' dis_loss : %.4f' % d_loss, ' gen_loss : %.4f' % g_loss)
        
        # 중간 과정을 이미지로 저장
        if i % saving_interval == 0 :
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
            
            # Rescale image 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            
            fig, axs = plt.subplots(5, 5)
            count = 0
            for j in range(5) :
                for k in range(5) :
                    axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[j, k].axis('off')
                    count += 1
            fig.savefig('./293p_gan_mnist/gan_mnist_%d.png' % i)

gan_train(4001, 32, 200) # 4000번 반복되고(+1 에 주의), 배치 사이즈는 32, 200번 마다 결과 저장
        
         
    