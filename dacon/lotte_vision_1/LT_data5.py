import numpy as np
import pandas as pd
import os
import cv2 as cv
from tensorflow.keras.utils import to_categorical

# INTER_NEAREST-최근 접 이웃 보간
# INTER_LINEAR-쌍 선형 보간 (기본적으로 사용됨)
# INTER_AREA-픽셀 영역 관계를 사용한 리샘플링. 무아레없는 결과를 제공하므로 이미지 데시 메이션에 선호되는 방법 일 수 있습니다. 그러나 이미지를 확대하면 INTER_NEAREST 방법과 유사합니다.
# INTER_CUBIC-4x4 픽셀 이웃에 대한 쌍 입방 보간
# INTER_LANCZOS4-8x8 픽셀 이웃에 대한 Lanczos 보간

# train_img_arr = []
# train_y = []
# print(train_y)
# for i in range(1000) :
#     for j in range(48) :
#         path = '../../data/LPD_competition/train/' + str(i) + '/' + str(j) + '.jpg'
#         print(path)
#         image = cv.imread(path)
#         image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)

#         train_img_arr.append(image)
#         train_y.append(i)

# train_img_arr = np.asarray(train_img_arr)
# train_y = np.asarray(train_y)
# train_y = train_y.reshape(-1, 1)
# train_y = to_categorical(train_y)

# print(train_img_arr.shape)
# print(train_y.shape)

# np.save('../../data/npy/train_x_224_gray.npy', arr = train_img_arr)
# np.save('../../data/npy/train_y_224_gray.npy', arr = train_y)

#===
test_image_arr = []
for i in range(72000) :
    path = '../../data/LPD_competition/test/' + str(i) + '.jpg'
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)
    test_image_arr.append(image)
    print(i)
test_image_arr = np.asarray(test_image_arr)

np.save('../../data/npy/predict_x_224_gray', arr = test_image_arr)
    