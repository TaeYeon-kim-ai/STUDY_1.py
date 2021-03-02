import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x = preprocess_input(x_train, x_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state= 128)

x_train = x_train.reshape(-1, 32*32*3)
x_test = x_test.reshape(-1, 32*32*3)
x_val = x_val.reshape(-1, 32*32*3)
print(x_train.shape, x_test.shape, x_val.shape)
#(40000, 1024, 3) (10000, 1024, 3) (10000, 1024, 3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#reshape
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
x_val = x_val.reshape(-1, 32, 32, 3)
print(x_train.shape, x_test.shape, x_val.shape)

#2. 모델링
TF = InceptionResNetV2(weights= 'imagenet', include_top = False, input_shape = (32, 32, 3)) #레이어 16개
TF.trainable = False #훈련시키지 않고 가중치만 가져오겠다.
model = Sequential()
model.add(TF)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) #activation='softmax')) #mnist사용할 경우
model.summary()
print(len(TF.weights)) # 26
print(len(TF.trainable_weights)) # 0

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [es])

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

#cifar10
# loss :  [3.007906913757324, 0.10000000149011612]
# [3 8 8 0 6 6 1 6 3 1]
