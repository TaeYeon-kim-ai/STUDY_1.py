import cv2
import numpy as np
from matplotlib import pyplot as plt
#이미지 넘파이로

#train
aaa = []

for i in range(50000):
    image_path = 'C:/data/vision_2/dirty_mnist_2nd_noise_clean/%05d.png'%i
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image1 = np.where((image <= 254) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
    image1 = cv2.dilate(image1, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image1 = cv2.medianBlur(src=image1, ksize= 5)
    image1 = cv2.resize(image1, (128, 128))
    image1 = np.asarray(image1).reshape(128, 128, 1)
    aaa.append(image1)

aaa = np.array(aaa)/255.

np.save('C:/data/vision_2/dirty_mnist_npy/x_train.npy', arr = aaa)
print(aaa.shape)


#test
bbb = []

for i in range(50000,55000):
    image_path = 'C:/data/vision_2/test_dirty_mnist_2nd_noise_clean/%05d.png'%i
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image1 = np.where((image <= 254) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
    image1 = cv2.dilate(image1, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image1 = cv2.medianBlur(src=image1, ksize= 5)
    image1 = cv2.resize(image1, (128, 128))
    image1 = np.asarray(image1).reshape(128, 128, 1)
    bbb.append(image1)

bbb = np.array(bbb)/255.

np.save('C:/data/vision_2/dirty_mnist_npy/x_test.npy', arr = bbb)
print(bbb.shape)
# cv2.waitKey(0)

