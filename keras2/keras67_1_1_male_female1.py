#실습
#남자 여자 구분
#ImageDataGenerator의 fit사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#http://naver.me/5B1Y91UT
#이미지 / 데이터로 전처리 / 증폭가능하게 //

#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255, #전처리 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=[-1, 1],
    height_shift_range=[-1, 1],
    rotation_range=3,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255) 

#데이터 .npy로 전환
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/gender',
    target_size = (64, 64),
    batch_size = 2000,
    class_mode= 'binary' 
)


print(xy_train)
#npy 저장
# np.save('C:/data/image/gender/npy/keras67_train_x.npy', arr=xy_train[0][0]) #x 1
# np.save('C:/data/image/gender/npy/keras67_train_y.npy', arr=xy_train[0][1])