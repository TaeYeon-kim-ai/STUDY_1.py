import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1.데이터
data = np.load('./stock_pred/SSD_prepro_data2.npy')

# #0. 함수정의
def split_x(seq, size, col):
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size),0:col]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 5 #며칠
col = 11 #열 수
dataset = split_x(data,size, col)

x = dataset[:-1,:5,0:10]
y = dataset[1:,0,-1]
x_pred = dataset[-1:,:5,0:10] 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state = 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle= True, random_state = 100)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
y_val = y_val.reshape(y_val.shape[0],1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred=scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], 5, 10)
x_test = x_test.reshape(x_test.shape[0], 5, 10)
x_val = x_val.reshape(x_val.shape[0], 5, 10)
x_pred=x_pred.reshape(1 , 5 ,10)

model = load_model('./stock_pred/Stock_SSD_2_model2.h5')#모델로드
#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss : ', loss)
print('mae : ', mae)
y_predict = model.predict(x_test)
x_predict = model.predict(x_pred)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

print('주가 :', x_predict)