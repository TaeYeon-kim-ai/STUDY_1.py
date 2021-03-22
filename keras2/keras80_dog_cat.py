#이미지는
#data/image/vgg/에 4개를 넣으시오
#개, 고양이, 라이언, 슈트
#이렇게 넣을 것
#파일명 : 
#dog1.jpg, cat1.jpg, lion.jpg, suit1.jpg
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#1. DATA
img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224, 224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224, 224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size=(224, 224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224, 224))
print(img_suit)
# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
print(arr_dog)
print(type(arr_dog)) #<class 'numpy.ndarray'>
print(arr_dog.shape) #(224, 224, 3)

#RGB => BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
#vgg16에 맞춰서 이미지 변환 #RGB => BGR
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog)
print(arr_dog.shape) #(224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
print(arr_input.shape) #(4, 224, 224, 3)

#2. MODEL
model = VGG16()
results = model.predict(arr_input)
print(results)
print('results.shape : ', results.shape) #results.shape :  (4, 1000) : imagenet에서 지급하는 category 수


# IMAGE RESULT
from tensorflow.keras.applications.vgg16 import decode_predictions #결과를 판단하겠다

decode_results = decode_predictions(results)
print("======================================================")
print("decode_results[0] : ", decode_results[0])
print("======================================================")
print("decode_results[1] : ", decode_results[1])
print("======================================================")
print("decode_results[2] : ", decode_results[2])
print("======================================================")
print("decode_results[3] : ", decode_results[3])
print("======================================================")
