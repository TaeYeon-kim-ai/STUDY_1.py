# 실습
# cifar10를 flow로 구성해서 완성
# ImageDataGenerator

import numpy as mp
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import  Sequential, Model
from tensorflow.keras.layers import  Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, Input, Dropout,BatchNormalization
from sklearn.model_selection import  train_test_split, KFold
from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import  to_categorical

#1. 데이터셋
#1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#1.1 데이터 전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=100)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rescale= 1./255, #전처리 
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=[-1, 1],
    height_shift_range=[-1, 1],
    rotation_range=3,
    zoom_range=0.3,
    # shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255) 

xy_train = datagen.flow(x_train, y=y_train, batch_size=16)
xy_test = test_datagen.flow(x_test, y=y_test, batch_size=16)
xy_val = test_datagen.flow(x_val, y=y_val, batch_size=16)


print(y_train.shape) 
print(x_train.shape)
# (40000, 10)
# (40000, 3072)

#2. 모델링
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = (32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3)))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
#from tensorflow.keras.callbacks import EarlyStopping
#es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit_generator(xy_train,
          steps_per_epoch= (len(xy_train)/16), epochs=50,
          validation_data=(xy_val),
          validation_steps=(len(xy_val)/16))

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print('loss : ', loss)
print('acc : ', acc)




















