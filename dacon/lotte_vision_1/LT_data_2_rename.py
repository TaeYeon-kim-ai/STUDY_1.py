import numpy as np
import cv2 as cv
import glob
from tensorflow.keras.utils import to_categorical
import random
import os

i = 0 #1부터 변경할 것임
for i in range(1000):
    path = '../../data/LPD_competition/train/' + str(i) + '/'
    file_names = os.listdir(path)
    for name in file_names: 
        src = os.path.join(path, name)
        dst = str(i) + '.jpg'
        dst = os.path.join(path, dst)
        os.rename(src, dst)
        i += 1