import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
path = r'C:/data/vision_2/dirty_mnist_2nd' # Source Folder
dstpath = r'C:/data/vision_2/dirty_mnist_2nd_noise_clean' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    #ret, addWeight = cv2.threshold(img, 255, 255, cv2.THRESH_TRUNC)
    img = np.where((img <= 252) & (img != 0), 0, img)
    img = np.where((img >= 128) & (img != 0), 247, img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    #결과 출력
    #cv2.imshow('THRESH_BINARY_INV', addWeight)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),img)#datetime A combination of a date and a time. Attributes: ()


path = r'C:/data/vision_2/test_dirty_mnist_2nd' # Source Folder
dstpath = r'C:/data/vision_2/test_dirty_mnist_2nd_noise_clean' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in asme folder")
# Folder won't used
files = os.listdir(path)

for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    #ret, addWeight = cv2.threshold(img, 255, 255, cv2.THRESH_TRUNC)
    img = np.where((img <= 252) & (img != 0), 0, img)
    img = np.where((img >= 128) & (img != 0), 247, img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    #결과 출력
    #cv2.imshow('THRESH_BINARY_INV', addWeight)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(dstpath,image),img)#datetime A combination of a date and a time. Attributes: ()
