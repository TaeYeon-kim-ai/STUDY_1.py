import tensorflow
import numpy as np
import glob
import cv2

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization,\
    Activation, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB7

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from PIL import Image

datagen = ImageDataGenerator(
    vertical_flip=True,
    rescale=1./255,
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1)
)

datagen2 = ImageDataGenerator(
    rescale=1./255
)

xy_data = datagen.flow_from_directory(
    directory='../../data/LPD_competition/train/',
    target_size=(256, 256),
    batch_size=100000,
    class_mode='categorical'
)

np.save('../../data/npy/train_x_256.npy', arr = xy_data[0][0])
np.save('../../data/npy/train_y_256.npy', arr = xy_data[0][1])

x_train = np.load('../../data/npy/train_x_256.npy')
y_train = np.load('../../data/npy/train_y_256.npy')

print(x_train.shape)
print(y_train.shape)

#===========================================================

pred = list()
for i in range(72000):
    img = cv2.imread(
        '../../data/LPD_competition/test/%s.jpg'%i
    )
    img = cv2.resize(img, (256, 256))
    img = np.array(img)
    pred.append(img)

pred = np.array(pred)

np.save('../../data/npy/predict_x_256.npy', arr = pred)

print(pred.shape)