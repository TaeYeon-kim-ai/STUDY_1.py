import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
data = np.load('./stock_pred/SSD_prepro_data2.npy')

# #0. 함수정의
def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size),0:col]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 5 #며칠
col = 11 #열 수
dataset = split_x(data,size, col)

# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2398 non-null   float64
#  1   고가      2398 non-null   float64
#  2   저가      2398 non-null   float64
#  3   거래량     2398 non-null   float64
#  4   금액(백만)  2398 non-null   float64
#  5   개인      2398 non-null   int64
#  6   기관      2398 non-null   int64
#  7   외인(수량)  2398 non-null   int64
#  8   외국계     2398 non-null   int64
#  9   외인비     2398 non-null   float64
#  10  종가      2398 non-null   float64
# dtypes: float64(7), int64(4)
x = dataset[:-1,:5,0:10]
y = dataset[1:,0,-1]
x_pred = dataset[-1:,:5,0:10] 
print(x.shape, y.shape, x_pred.shape)
# (2393, 5, 10) (2393, 5) (1, 5, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state = 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle= True, random_state = 100)

#(1531, 5) (479, 5) (383, 5)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
y_val = y_val.reshape(y_val.shape[0],1)

print(x_train.shape) # (1529, 5, 10)
print(x_test.shape) # (478, 5, 10)
print(x_val.shape) # (383, 5, 10)

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
print(x_pred)

# 2. 모델링
input1 = Input(shape=(x.shape[1],x.shape[2]))
lstm = LSTM(200,activation='relu',input_shape = (x.shape[1], x.shape[2]))(input1)
drop = Dropout(0.2)(lstm)
dense1 = Dense(150, activation='relu')(drop)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(90, activation='relu')(dense1)
dense1 = Dense(70, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1) 
outputs = Dense(1, activation= 'relu')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()
# model.save('./stock_pred/Stock_SSD_2_model1.h5')#모델저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = './stock_pred/MCP/Stock_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 25, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 16, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping]) #cp])

# model.save('./stock_pred/Stock_SSD_2_model2.h5') #모델저장2
# model.save_weights('./stock_pred/Stock_SSD2_weight.h5') #weight저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드
# model.load_weights('./stock_pred/k52_1_weight.h5') #weight로드

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

# # 시각화
# plt.rc('font', family='Malgun Gothic')
# plt.figure(figsize=(10,6))
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
# plt.grid()
# plt.title('cost_loss') #loss, cost
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.subplot(2,1,2)
# plt.plot(hist.history['mae'], marker = '.', c = 'red', label = 'mae')
# plt.plot(hist.history['val_mae'], marker = '.', c = 'blue', label = 'val_mae')
# plt.grid()
# plt.title('cost_mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')
# plt.show()

# loss :  428164.28125
# mae :  484.46746826171875
# RMSE:  654.3425938329358
# R2:  0.9975475692123729
# 주가 : [90037.07]

# loss :  410697.96875
# mae :  469.6899719238281
# RMSE:  640.8572746275237
# R2:  0.9976476116809886
# 주가 : [[89991.836]]