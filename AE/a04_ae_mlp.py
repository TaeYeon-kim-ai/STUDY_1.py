#2번카피 복붙
#딥러닝 모델로 구성
#autoencoder규칙
#1번 : 784, 784
# 2개의 모델을 만들기, 하나는 원칙적 오토인코더 대칭
# 다른 하나는 랜덤하게 만들고 싶은 대로 히든을 구성
# 2개의 성능 비교

import numpy as np 
from tensorflow.keras.datasets import mnist

#(x_train, _), (x_test, _) = mnist.load_data() #y를 로드 후 사용하지 않아도 되지만 낭비임.
(x_train, _), (x_test, _) = mnist.load_data() #y를 로드하지 않음(비지도이므로 X만 활용)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#autoencoder model
def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, activation='relu', input_shape = (784,)))
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dense(units = 128, activation='relu'))
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

#deeplearning model
'''
def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, activation='relu', input_shape = (784,)))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(128))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model
'''

model = autoencoder(hidden_layer_size = 154)

#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics='acc')
model.compile(optimizer = 'adam', loss = 'mse', metrics='acc')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize = (20, 7))

#이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토 인코더로 출력된 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()





