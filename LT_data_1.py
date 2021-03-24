# 10개의 판별 모델을 만들어서 competition 벌이는 대환장 소스코드!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from natsort import natsorted
import cv2 as cv
import numpy as np
import os
import gc

#########################
TRAIN_DIR = '../../data/LPD_competition/train'
TEST_DIR = '../../data/LPD_competition/test'

DIMENSION = 256
atom = 10

train_fnames = natsorted(os.listdir(TRAIN_DIR)) # 1000
test_fnames = natsorted(os.listdir(TEST_DIR))

# 제너레이터 정의 (Initialize image data generator)
########################
train_datagen = ImageDataGenerator(
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range=5,
    zoom_range=0.2,
)

test_datagen = ImageDataGenerator()
########################

for idx, folder in enumerate(train_fnames): # 트레인 폴더는 1000개 있다
    # if idx >= 1:
    #     break
    print("folder :", folder)
    base_dir = TRAIN_DIR + '/' + folder + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/train/0'
    print("basedir :", base_dir)
    img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트
    print(img_lst)

    for i, f in enumerate(img_lst): # 각 폴더에 접근해서
        # if i >= 1:
        #     break
        print("f :", f)
        img_dir = base_dir + f # 'C:/data/LPD_competition/train/0/0.jpg'
        print("img_dir :", img_dir)
        img = np.expand_dims(image.load_img(img_dir, target_size=(DIMENSION, DIMENSION)), axis=0) # 각 이미지를 불어온다
        print(img, "\n", img.shape)
        # cv_img = cv.imread(img_dir)
        # cv.imshow("whatever", cv_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        train_datagen.fit(img) #image gen
        for x, val in zip(train_datagen.flow(x=img,
            save_to_dir=base_dir,
            save_prefix='aug',
            shuffle=False), range(1)) :
            pass
        print("base dir :", base_dir)