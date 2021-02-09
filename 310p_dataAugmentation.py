# 201117_이 코드는 동작하지 않음. 아래 상태에서 멈춤. -> steps_per_epoch = 32 로 수정하면 동작함
'''
32/100 [========>.....................] - ETA: 4s - loss: 0.7104 - accuracy: 0.4437
WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or
generator can generate at least `steps_per_epoch * epochs` batches (in this case, 2000 batches).
You may need to use the repeat() function when building your dataset.
32/100 [========>.....................] - 2s 67ms/step - loss: 0.7104 - accuracy: 0.4437 - val_loss: 0.6921 - val_accuracy: 0.6500
'''
 

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers


# 마찬가지로 위, 아래로 이동
train_datagen = ImageDataGenerator(rescale = 1./255, # 정규화
        horizontal_flip = True, # 수평 대칭 이미지를 50% 확률로 만들어 추가
        width_shift_range = 0.1, # 전체 크기의 10 % 범위에서 좌우로 이동 
        height_shift_range = 0.1, # 전체 크기의 10% 범위에서 상하로 이동
        #rotation_range=5,
        #shear_range=0.7,
        #zoom_range=[0.9, 2.2],
        #vertical_flip=True,
        fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory('./dataset/train_310p', 
        target_size = (150, 150), batch_size = 5, class_mode = 'binary')

# 테스트셋은 이미지 부풀리기 과정 미진행
test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory('./dataset/test_310p', target_size = (150, 150), 
        batch_size = 5, class_mode = 'binary')

# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# 모델 컴파일
model.compile(loss = 'sparse_categorical_crossentropy', 
        optimizer = optimizers.Adam(learning_rate = 0.0002), metrics = ['accuracy'])

# 모델 실행
history = model.fit_generator(train_generator, steps_per_epoch = 32, epochs = 20, 
        validation_data = test_generator, validation_steps = 4)

# 결과를 그래프로 표현
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker = '.', c = 'red', label = 'Trainset_acc')
plt.plot(x_len, val_acc, marker = '.', c = 'lightcoral', label = 'Testset_acc')
plt.plot(x_len, y_vloss, marker = '.', c = 'cornflowerblue', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'trainset_loss')

plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss / acc')
plt.show()
