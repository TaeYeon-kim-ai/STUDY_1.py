import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# size_test2
path = r'C:/data/fish_data/test_image2/test/train1_fish_normal_2229' # Source Folder
dstpath = r'C:/data/fish_data/test_image/test/train2_fish_illness_556' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    image1 = cv2.resize(img, (240, 160), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(dstpath,image),image1)