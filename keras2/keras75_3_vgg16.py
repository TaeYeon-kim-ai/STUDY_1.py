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
model.add(Dense(1)) #activation='softmax')) #mnist사용할 경우
model.summary()

print("동결 전 훈련되는 가중치의 수 : ", len(model.weights)) #32
print("동결한 후 훈련되는 가중치의 수 : ", len(model.trainable_weights)) # 6 

####################### 하단 추가 ###############################
import pandas as pd
pd.set_option('max_colwidth', -1) 
layers = [(layer, layer.name, layer.trainable) for layer in model.layers] #모델에 대해 반환되는 값을 모델에 넣고 그걸 데이터프레임에 저장하겠다.
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable']) #컬럼명은 ~~ ~~ ~~

print(aaa)
'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000025139439370>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x000002513F4B2E50>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x000002513F4F9AC0>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x000002513F4F9DC0>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x000002513681C6A0>             dense_2    True
'''