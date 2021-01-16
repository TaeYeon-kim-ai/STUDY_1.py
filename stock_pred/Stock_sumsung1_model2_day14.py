import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1.데이터
data = np.load('./stock_pred/SSD_prepro_data1.npy')

# #0. 함수정의
def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size)]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 9 #며칠
col = 10 #열 수
dataset = split_x(data,size, col)

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2397 non-null   float64
#  1   고가      2397 non-null   float64
#  2   저가      2397 non-null   float64
#  3   거래량     2397 non-null   float64
#  4   금액(백만)  2397 non-null   float64
#  5   개인      2397 non-null   float64
#  6   기관      2397 non-null   float64
#  7   외국계     2397 non-null   float64
#  8   프로그램    2397 non-null   float64
#  9   종가      2397 non-null   float64
# dtypes: float64(10)
# memory usage: 206.0+ KB

x = dataset[:, :, 9] #(2388, 9)
y = dataset[:,:1,-1:] # (2388, 1, 1)
x_pred = dataset[-1:,:,:] #(1, 9, 10)
# x = dataset[:-1, :-1] # 컬럼값
# y = dataset[1:, :] # 종가값
# x_pred = dataset[-1:, :-1] # 예측값
print(x_pred)
print(x.shape, y.shape, x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state = 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle= True, random_state = 100)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


# 2. 모델링
input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(200,activation='relu', input_shape = (x.shape[1], 1))(input1)
drop = Dropout(0.20)(lstm)
dense1 = Dense(128, activation='relu')(drop)
drop = Dropout(0.25)(dense1)
dense1 = Dense(128, activation='relu')(drop)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1) 
outputs = Dense(1, activation= 'relu')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()
# model.save('../data/h5/Stock_SSD_1_model1.h5')#모델저장
# model = load_model('../data/h5/k51_1_model1.h5')#모델로드

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/Stock_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 16, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

# model.save('../data/h5/Stock_SSD_1_model2.h5') #모델저장2
# model.save_weights('../data/h5/Stock_SSD_weight.h5') #weight저장
# model = load_model('../data/h5/k51_1_model1.h5')#모델로드
# model.load_weights('../data/h5/k52_1_weight.h5') #weight로드

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size= 7)
print('loss : ', loss)
print('mae : ', mae)
y_predict = model.predict(x_test)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

# x_pred = scaler.transform(x_pred)
# x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1],x_pred.shape[2])
predict = model.predict(x_pred)
print('주가 :', predict[-1])


# x_predict = np.array([[89800,91200,89100,4557102,-1781416, 55.56]])
# x_predict = scaler.transform(x_predict)
# x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1) # 3차원 
# y_predict = model.predict(x_predict)


# # 시각화
# import matplotlib.pyplot as plt
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
