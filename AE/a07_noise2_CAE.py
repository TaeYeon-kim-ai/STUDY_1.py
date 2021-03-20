#[과제] Conv2D


#[ 실습 ] CNN으로 레이어 autoencoder구성
import numpy as np 
from tensorflow.keras.datasets import mnist

#(x_train, _), (x_test, _) = mnist.load_data() #y를 로드 후 사용하지 않아도 되지만 낭비임.
(x_train, _), (x_test, _) = mnist.load_data() #y를 로드하지 않음(비지도이므로 X만 활용)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.


#랜덤하게 노이즈(점) 찍으나, 0 ~ 1사이 값을 가지는 곳에 0 ~ 0.1의 값을 찍게 되면 1의 값에는 0.1이 더해질 경우 최대치가 1.1이 되므로
#다시 1로 반환할 필요가 있다.
#따라서 np.clip을 통해 a_max = 1로 적용하여 1이 넘어가는 수치를 1로 반환해준다.
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape) # 랜덤하게 점 찍기 
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape) # 랜더덤하게 점 찍기
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1) 
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)


#MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input ,Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Flatten, BatchNormalization, ReLU, LeakyReLU, UpSampling2D
def autoencoder(hidden_layer_size): 
    inputs = Input(shape = (28, 28, 1))
    x = Conv2D(filters = 128, kernel_size= 2, strides=2, padding="SAME")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = 64, kernel_size= 2, strides=2, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x1 = x

    x = Conv2DTranspose(filters = 128, kernel_size= 2, strides=2, padding="SAME")(x+x1)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D()(x)

    x = Dense(1, activation='sigmoid')(x) #흑백 1, 컬러3
    outputs = x
    model = Model(inputs = inputs, outputs = outputs)
    
    model.summary()
    return model
    
model = autoencoder(hidden_layer_size = 154) #95% 적용, 가장 안정적인 수치 출력하기 위함

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10)
output = model.predict(x_test_noised) #x_test노이즈가 제가되었는지 확인

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), 
    (ax6, ax7, ax8, ax9, ax10), 
    (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))

#이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

#원본 입력 이미지를 맨 위에 그린다!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("NOISE", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()