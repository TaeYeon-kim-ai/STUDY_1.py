# autoencoder 비지도학습(unsupervised Learning)
# X에 대한 목표값 Y가 없음.
# X와 Y가 동의 X----X
# PCA, ml_tree구조 등
# 준 지도 학습이라고도 함
# 1    2    3    4
# ㅇ        ㅇ     
# ㅇ   ㅇ   ㅇ   ㅇ
# ㅇ   ㅇ   ㅇ   ㅇ    
# ㅇ   ㅇ   ㅇ   ㅇ
# ㅇ   ㅇ   ㅇ   ㅇ
# ㅇ        ㅇ
# 1~2단계 까지의 훈련과정을 거치면서 잡음이 제거됨
# 3에서 데이터 증폭
# 예) 이미지원복, Gan

#input_dim 5
#output_dim 3 제거하고 확장하는 과정에서 다 제거됨
#input_dim 3
#output_dim 5

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

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) #모델
decoded = Dense(784, activation='sigmoid')(encoded)
#decoded = Dense(784, activation='relu')(encoded) #0이하의 수를 0으로 반환하기에 하단부분의 데이터가 다 깍여나감

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc'])
#autoencoder.compile(optimizer='adam', loss = 'mse')

autoencoder.fit(x_train, x_train, epochs = 30, batch_size = 128, validation_split = 0.2)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n) : 
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28)) #원래 x_test의 이미지 출력하고
    plt.gray()
    ax.get_xaxis().set_visible(False) #옆에있는 숫자 제거
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28)) #decoding된 이미지 10개 출력
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()