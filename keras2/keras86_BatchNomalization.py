#다 비슷한데 약간의 차이가 있음.
#[ 과 제 ]
# kernel_initialier  .... He     relu friends?
#                    .... Xavier sigmoid, tanh?
# vias_initiallizer  
# kernel_regularizer 

# BatchNormalization 
# Dropout ... BatchNormal과 같이 쓰지 않는다(통상) 
# 코드에 중복적인 부분이 있어서 오히려 안좋아지는 부분이 있을 수 있다??

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

#1. 데이터
(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #1생략 흑백
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,) #1생략 흑백

print(x_train[0])
print(y_train[0])

print("y_trian[0] : ", y_train[0])
print(x_train[0].shape) #(28, 28)

# #이미지 보기
# #plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

#1.1 데이터 전처리
#데이터 전처리를 해야함(Min, Max)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
x_test = x_test.reshape(10000, 28, 28, 1)/255.
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.max)
print(x_train.min)

#sklearn.onehotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
# model.add(Dropout(0.2))

model.add(Conv2D(200, kernel_initializer="he_normal"))
model.add(BatchNormalization()) #위에 노멀에 대해 정규화 하겠다.
model.add(Activation('relu'))

model.add(Conv2D(100, kernel_regularizer=l1(0.01))) #l1은 정규화 regularizer... 정직화, 정규화/? Minmax(정규화), standard(일반화, 표준화)
model.add(Dropout(0.2))                             #l1 노른 relu계열 .... xegumal // tan계열 ,, he_normal

model.add(Conv2D(100, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(16, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
#filepath='(경로)' : 가중치를 세이브 해주는 루트
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

