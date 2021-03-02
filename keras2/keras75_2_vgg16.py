from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16


#include_top True로 할경우 imagent에 세팅된것 그대로 가져옴
#vgg16 defult = 244, 244, 16
vgg16 = VGG16(weights= 'imagenet', include_top = False, input_shape = (32, 32, 3)) #레이어 16개
#print(vgg16.weights)

vgg16.trainable = False #훈련시키지 않고 가중치만 가져오겠다.
vgg16.summary()
print(len(vgg16.weights)) # 26
print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #activation='softmax'))
model.summary()

print("동결 전 훈련되는 가중치의 수 : ", len(model.weights)) #32
print("동결한 후 훈련되는 가중치의 수 : ", len(model.trainable_weights)) # 6 
'''
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
'''