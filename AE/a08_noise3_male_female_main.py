#주근깨 넣어서 여드름 제거

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#http://naver.me/5B1Y91UT
#이미지 / 데이터로 전처리 / 증폭가능하게 //

#train/test준비 선언
train_datagen = ImageDataGenerator(
    width_shift_range=[-1, 1],
    height_shift_range=[-1, 1],
    rotation_range=3,
    zoom_range=0.3,
    fill_mode='nearest'
)

#IDG_train
x_train = train_datagen.flow_from_directory(
    'C:/data/image/gender_2_noise_test/',
    target_size = (128, 128),
    batch_size = 4000,
)

#npy 저장
#np.save('C:/data/image/gender_npy/a08_noise_test_train_x.npy', arr=x_train[0][0])

#npy 로드
x_train = np.load('C:/data/image/gender_npy/a08_noise_test_train_x.npy')
print(x_train.shape) #(1736, 128, 128, 3)

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x_train, train_size = 0.8, random_state = 200, shuffle = True)
print(x_train.shape, x_test.shape) #(1388, 128, 128, 3) (348, 128, 128, 3)
x_train = x_train.reshape(1388, 128, 128, 3).astype('float32')/255
x_test = x_test.reshape(348, 128, 128, 3)/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape) # 랜덤하게 점 찍기 
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape) # 랜더덤하게 점 찍기
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1) 
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

#load_model
model = load_model('../data/h5/male_female_model2.h5')
output = model.predict(x_test_noised) #x_test노이즈가 제가되었는지 확인
model.load_weights('../data/h5/male_female__weight.h5')

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
    ax.imshow(x_test[random_images[i]].reshape(128, 128, 3), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(128, 128, 3), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("NOISE", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(128, 128, 3), cmap = 'gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
