import cv2 
import numpy as np

img = cv2.imread()

bgr = cv2.imread()

bgra = cv2.imread()

print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape)

cv2.imshow('bgr', bgr)