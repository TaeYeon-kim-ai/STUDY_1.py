import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
data1 = np.load('./stock_pred/SSD_prepro_data3.npy')
data2 = np.load('./stock_pred/SSD_prepro_ETF_data3.npy')

# #0. 함수정의
def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size),0:col]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 6 #며칠
col = 7 #열 수
dataset = split_x(data1,size, col)
dataset2 = split_x(data2,size, col)
print('dataset:', dataset)
print('dataset2:', dataset2)

# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   고가      1088 non-null   int64
#  1   저가      1088 non-null   int64
#  2   종가      1088 non-null   int64
#  3   거래량     1088 non-null   int64
#  4   금액(백만)  1088 non-null   int64
#  5   개인      1088 non-null   int64
#  6   시가      1088 non-null   int64
# dtypes: int64(7)

x1 = dataset[:-1,0:6,0:6]
x1_pred = dataset[-1:,0:6,0:6]
x2 = dataset2[:-1,0:6,0:6]
x2_pred = dataset2[-1:,0:6,0:6]
y = dataset[1:,0,-1] # OR [1:,-1:,0] 하루 건너 치, y=df[1:, -2:, 0]이틀치 from.주형
print(x1.shape)
print(y.shape)

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size = 0.8, shuffle=True, random_state = 100)
x1_train, x1_val, y_train, y_val = train_test_split(x1_train, y_train, train_size = 0.8, shuffle= True, random_state = 100)
x2_train, x2_test = train_test_split(x2, train_size = 0.8, shuffle=True, random_state = 100)
x2_train, x2_val= train_test_split(x2_train, train_size = 0.8, shuffle= True, random_state = 100)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x1_val = x1_val.reshape(x1_val.shape[0], x1_val.shape[1]*x1_val.shape[2])
x1_pred = x1_pred.reshape(x1_pred.shape[0], x1_pred.shape[1]*x1_pred.shape[2])

x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2])
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2])
x2_val = x2_val.reshape(x2_val.shape[0], x2_val.shape[1]*x2_val.shape[2])
x2_pred = x2_pred.reshape(x2_pred.shape[0], x2_pred.shape[1]*x2_pred.shape[2])

#1.1 전처리
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_val = scaler.transform(x1_val)
x1_pred=scaler.transform(x1_pred)

scaler2 = MinMaxScaler()
scaler2.fit(x2_train)
x2_train = scaler2.transform(x2_train)
x2_test = scaler2.transform(x2_test)
x2_val = scaler2.transform(x2_val)
x2_pred=scaler2.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0], 6, 6)
x1_test = x1_test.reshape(x1_test.shape[0], 6, 6)
x1_val = x1_val.reshape(x1_val.shape[0], 6, 6)
x1_pred=x1_pred.reshape(1 ,6,6) #pred값 수정

x2_train = x2_train.reshape(x2_train.shape[0], 6, 6)
x2_test = x2_test.reshape(x2_test.shape[0], 6, 6)
x2_val = x2_val.reshape(x2_val.shape[0], 6, 6)
x2_pred=x2_pred.reshape(1 ,6, 6) #pred값 수정

y_train = y_train.reshape(y_train.shape[0] ,1)
y_test = y_test.reshape(y_test.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

# 2. 모델링
# 모델1
input1 = Input(shape=(x1.shape[1],x1.shape[2]))
lstm1 = LSTM(250,activation='relu',input_shape = (x1.shape[1], x1.shape[2]))(input1)
drop1 = Dropout(0.1)(lstm1)
dense1 = Dense(150, activation='relu')(drop1)
dense1 = Dense(130, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1) 
dense1 = Dense(10, activation='relu')(dense1) 

#모델2
input2 = Input(shape=(x2.shape[1],x2.shape[2]))
lstm2 = LSTM(250,activation='relu',input_shape = (x2.shape[1], x2.shape[2]))(input2)
drop2 = Dropout(0.1)(lstm2)
dense2 = Dense(150, activation='relu')(drop2)
dense2 = Dense(130, activation='relu')(dense2)
dense2 = Dense(100, activation='relu')(dense2)
dense2 = Dense(64, activation='relu')(dense2) 
dense2 = Dense(10, activation='relu')(dense2) 

#모델 병합/ concatenate
merge = concatenate([dense1, dense2])
middle = Dense(72, activation= 'relu')(merge)
middle = Dense(32, activation= 'relu')(middle)
middle = Dense(32, activation= 'relu')(middle)
middle = Dense(32, activation= 'relu')(middle)
middle = Dense(16, activation= 'relu')(middle)
outputs = Dense(2)(middle)

#모델 선언
model = Model(inputs = [input1, input2], outputs = outputs)
model.summary()

model.save('./stock_pred/Stock_SSD_3_model1.h5')#모델저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드

print(x1_train.shape, x2_train.shape)#(692, 5, 5) (692, 5, 5)
print(y_train.shape) #(692, 1)

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './stock_pred/MCP/Stock_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor = 'loss', patience = 25, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit([x1_train, x2_train], y_train, epochs = 1000, batch_size = 18, validation_data = ([x1_val, x2_val], y_val), verbose = 1 ,callbacks = [early_stopping, cp, reduce_lr]) 

model.save('./stock_pred/Stock_SSD_3_model2.h5') #모델저장2
model.save_weights('./stock_pred/Stock_SSD3_weight.h5') #weight저장
# model = load_model('./stock_pred/k51_1_model1.h5')#모델로드
# model.load_weights('./stock_pred/k52_1_weight.h5') #weight로드

#4. 평가, 예측
loss, mae = model.evaluate([x1_test, x2_test], y_test ,batch_size=7)
y_predict = model.predict([x1_test, x2_test])
x_predict = model.predict([x1_pred, x2_pred])
print('loss : ', loss)
print('mae : ', mae)
y_predict = model.predict([x1_test, x2_test])
x_predict = model.predict([x1_pred,x2_pred])
print('18일 시가 :', x_predict[0])
print('19일 시가 :', x_predict[0,1:])

# #RMSE, R2
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE: ', RMSE(y_test, y_predict))

# from sklearn.metrics import r2_score
# R2 = r2_score(y_test, y_predict)
# print('R2: ', R2)

# 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()
plt.title('cost_loss') #loss, cost
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['mae'], marker = '.', c = 'red', label = 'mae')
plt.plot(hist.history['val_mae'], marker = '.', c = 'blue', label = 'val_mae')
plt.grid()
plt.title('cost_mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

#손대기 전..
# loss :  333201.21875
# mae :  424.3497619628906
# 18일 시가 : [87843.32 87613.21]
# 19일 시가 : [87613.21]

#일반
# loss :  753228.75
# mae :  736.1385498046875
# 18일 시가 : [91061.36 90785.85]
# 19일 시가 : [90785.85]

#reduce적용
# loss :  371473.8125
# mae :  452.0333557128906
# 18일 시가 : [85729.6   86740.125]
# 19일 시가 : [86740.125]
# loss :  273666.46875
# mae :  380.3132019042969
# 18일 시가 : [88710.125 88823.33 ]
# 19일 시가 : [88823.33]