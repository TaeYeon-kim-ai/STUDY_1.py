import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
data1 = np.load('./stock_pred/SSD_prepro_data2.npy')
data2 = np.load('./stock_pred/SSD_prepro_data3.npy')

# #0. 함수정의
def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size),0:col]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 10 #며칠
col = 11 #열 수
dataset = split_x(data1,size, col)
dataset = split_x(data2,size, col)

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
x1 = dataset[:-1,0:10,0:10]
y1 = dataset[1:,0,-1]
x1_pred = dataset[-1:,0:10,0:10]
x2 = dataset[:-1,0:10,0:10]
x2_pred = dataset[-1:,0:10,0:10]
print(x1.shape)
print(x2.shape)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.8, shuffle=True, random_state = 100)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size = 0.8, shuffle= True, random_state = 100)
x2_train, x2_test = train_test_split(x2, train_size = 0.8, shuffle=True, random_state = 100)
x2_train, x2_val= train_test_split(x2_train, train_size = 0.8, shuffle= True, random_state = 100)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x1_val = x1_val.reshape(x1_val.shape[0], x1_val.shape[1]*x1_val.shape[2])
x1_pred = x1_pred.reshape(x1_pred.shape[0], x1_pred.shape[1]*x1_pred.shape[2])

x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2])
x2_val = x2_val.reshape(x2_val.shape[0], x2_val.shape[1]*x2_val.shape[2])
x2_pred = x2_pred.reshape(x1_pred.shape[0], x2_pred.shape[1]*x2_pred.shape[2])

#1.1 전처리
scaler = MinMaxScaler()
scaler.fit(x1_train, x2_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_val = scaler.transform(x1_val)
x1_pred=scaler.transform(x1_pred)

x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_val = scaler.transform(x2_val)
x2_pred=scaler.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0], 10, 10)
x1_test = x1_test.reshape(x1_test.shape[0], 10, 10)
x1_val = x1_val.reshape(x1_val.shape[0], 10, 10)
x1_pred=x1_pred.reshape(1 , 10 ,10)

x2_train = x2_train.reshape(x2_train.shape[0], 10, 10)
x2_test = x2_test.reshape(x2_test.shape[0], 10, 10)
x2_val = x2_val.reshape(x2_val.shape[0], 10, 10)
x2_pred=x2_pred.reshape(1 , 10 ,10)

y1_train = y1_train.reshape(y1_train.shape[0] ,1)
y1_test = y1_test.reshape(y1_test.shape[0], 1)
y1_val = y1_val.reshape(y1_val.shape[0], 1)

# 2. 모델링
# 모델1
input1 = Input(shape=(x1.shape[1],x1.shape[2]))
lstm1 = LSTM(200,activation='relu',input_shape = (x1.shape[1], x1.shape[2]))(input1)
drop1 = Dropout(0.2)(lstm1)
dense1 = Dense(150, activation='relu')(drop1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(90, activation='relu')(dense1)
dense1 = Dense(70, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1) 
outputs1 = Dense(1, activation= 'relu')(dense1)

#모델2
input2 = Input(shape=(x2.shape[1],x2.shape[2]))
lstm2 = LSTM(200,activation='relu',input_shape = (x2.shape[1], x2.shape[2]))(input2)
drop2 = Dropout(0.2)(lstm2)
dense2 = Dense(150, activation='relu')(drop2)
dense2 = Dense(100, activation='relu')(dense2)
dense2 = Dense(100, activation='relu')(dense2)
dense2 = Dense(90, activation='relu')(dense2)
dense2 = Dense(70, activation='relu')(dense2)
dense2 = Dense(64, activation='relu')(dense2)
dense2 = Dense(32, activation='relu')(dense2)
dense2 = Dense(16, activation='relu')(dense2) 
outputs2 = Dense(1, activation= 'relu')(dense2)

#모델 병합/ concatenate
merge = concatenate([dense1, dense2])
middle = Dense(10)(merge)
middle = Dense(10)(middle)
middle = Dense(1)(middle)

#모델 선언
model = Model(inputs = [input1, input2], outputs = [outputs1, outputs2])
model.summary()

# model.save('./stock_pred/Stock_SSD_2_model1.h5')#모델저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드

print(x1_train.shape, x2_train.shape)#(1530, 5, 10) (1224, 5, 10)
print(y1_train.shape) #(1530, 1)

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = './stock_pred/MCP/Stock_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 25, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit([x1_train, x2_train], y1_train, epochs = 1000, batch_size = 32, validation_data = ([x1_val, x2_val], y1_val), verbose = 1 ,callbacks = [early_stopping]) #cp])

# model.save('./stock_pred/Stock_SSD_2_model2.h5') #모델저장2
# model.save_weights('./stock_pred/Stock_SSD2_weight.h5') #weight저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드
# model.load_weights('./stock_pred/k52_1_weight.h5') #weight로드

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test ,batch_size=7)
print('loss : ', loss)
y_predict = model.predict([x1_test, x2_test])
x_predict = model.predict(x1_pred)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y1_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y1_test, y_predict)
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

#15일
# loss :  410697.96875
# mae :  469.6899719238281
# RMSE:  640.8572746275237
# R2:  0.9976476116809886
# 주가 : [[89991.836]]