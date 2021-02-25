import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(50000):
    image_path = 'C:/data/vision_2/dirty_mnist_2nd/%05d.png'%i
    image = cv2.imread(image_path)
    image2 = np.where((image <= 254) & (image != 0), 0, image)#254보다 작은건 모조리 0으로 처리
    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image4 = cv2.medianBlur(src=image3, ksize= 5)
    cv2.imwrite('C:/data/vision_2/dirty_mnist_2nd_noise_clean/%05d.png'%i, image4)

for i in range(50000,55000):
    image_path = 'C:/data/vision_2/test_dirty_mnist_2nd/%05d.png'%i
    image = cv2.imread(image_path)
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image4 = cv2.medianBlur(src=image3, ksize= 5)
    cv2.imwrite('C:/data/vision_2/test_dirty_mnist_2nd_noise_clean/%05d.png'%i, image4)


# cv2.waitKey(0)