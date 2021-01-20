#자르기 - 2일 골라내기 - 예측 /// 7일될때마다 2개씩 골라내기

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
train = np.load('./dacon1/data/test/sun_train_data.npy')
test = np.load('./dacon1/data/test/sun_test_data.npy')\









# 1.1 비교 데이터 추출
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps            
        y_end_number = x_end_number + y_column
    
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i : x_end_number, :]
        tmp_y = dataset[x_end_number : y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(train, 47*7, 47*2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 0)

print(x_train.shape) #(33368, 47, 7, 8)
print(y_train.shape) #(33368, 47, 2, 8)

'''
#2. 모델링
input1 = Input(shape = (x.shape[1], x.shape[2]))
lstm = LSTM(50, activation='relu', input_shape = (x.shape[1], x.shape[2]))(input1)
dense1 = Dense(64, activation='relu')(lstm)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
outputs = Dense(1, activation='relu')(dense1)
#모델 선언
model = Model(inputs = input1, outputs = outputs)

#3컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './dacon1/data/MCP/sun_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
es = EarlyStopping(monitor='loss', patience=25, mode = 'auto')
model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data = (x_val, y_val), verbose=1, callbacks=[es, cp, reduce_lr])
#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=7)
y_predict = model.predict(x_test)
# x_predict = model.predict(y_test)
print('loss : ', loss)
print('mae : ', mae)
print('result : ', y_predict)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)
'''