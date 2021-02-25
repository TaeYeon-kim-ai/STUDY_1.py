import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.eager.monitoring import Sampler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

#1. 데이터

#ImageDatagenrtator & data augmentation
#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.05,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255) 

#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/vision_2/dirty_mnist_2nd_noise_clean',
    target_size = (256, 256), 
    batch_size = 50000, 
    class_mode = 'binary',
    subset="training"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/vision_2/test_dirty_mnist_2nd_noise_clean',
    target_size = (256, 256),
    batch_size = 5000, 
    class_mode = 'binary'
)

#npy 저장하기
np.save('C:/data/vision_2/dirty_mnist_npy/vision2_train_x.npy', arr=xy_train[0][0]) #x 1
np.save('C:/data/vision_2/dirty_mnist_npy/vision2_train_y.npy', arr=xy_train[0][1])
np.save('C:/data/vision_2/dirty_mnist_npy/vision2_test_x.npy', arr=xy_test[0][0])
np.save('C:/data/vision_2/dirty_mnist_npy/vision2_test_y.npy', arr=xy_test[0][1])