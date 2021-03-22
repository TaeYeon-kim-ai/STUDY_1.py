import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드

caltech_dir =  '../../data/LPD_competition/train/' #폴더 경로지정
categories = [] 
for i in range(0,1000) : #1000개  0부터 숫자 세서  i 안에 넣기
    i = "%d"%i #i를 정수로해서 i에 추가
    categories.append(i) #categories에 i를 리스트로 넣기

nb_classes = len(categories) #nb_classes는 

image_w = 256
image_h = 256

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_
        .w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 1000 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("../../data/npy/LPD_train_x_256.npy", arr=X)
np.save("../../data/npy/LPD_train_y_256.npy", arr=y)
# x_pred = np.load("../data/npy/P_project_test.npy",allow_pickle=True)
x = np.load("../../data/npy/LPD_train_x_256.npy",allow_pickle=True)
y = np.load("../../data/npy/LPD_train_y_256.npy",allow_pickle=True)

print(x.shape)
print(y.shape)