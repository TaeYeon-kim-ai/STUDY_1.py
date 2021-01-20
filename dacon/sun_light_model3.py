#자르기 - 2일 골라내기 - 예측 /// 7일될때마다 2개씩 골라내기

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
x_train = pd.read_csv('./dacon/data/train/train.csv')

# 1.1 비교 데이터 추출 
def preprocess_data(data):
        temp = data.copy()                                                  #temp를 정의
        return temp.iloc[-48:,:]

df_test = []

#test data병합
for i in range(81): # 갯수 지정
    file_path = './dacon/data/test/' + str(i) + '.csv'                      #test파일 81개를 함수를 통해 병합함
    temp = pd.read_csv(file_path)                                           #파일을 읽어드린 후  temp로 지정
    temp = preprocess_data(temp)                                            #temp파일 처리
    df_test.append(temp)                                                    #temp파일을 df_test에 추가함.

x_pred = pd.concat(df_test)                                                 #
# Attach padding dummy time series
x_pred = x_pred.append(x_pred[-96:])                                        #적용 안하면 (3888, 9) x_test  의 예측을 위한 공간확보(?)
print(x_pred.shape) #(3984, 9)                                                                
print(x_pred.shape) #(3888, 9)                                              #적용 안하면 (3888, 9) x_test  의 예측을 위한 공간확보(?)

df_train = x_train.iloc[:, 3:]                                              #Minute으로 비교
df_test = x_pred.iloc[:, 3:]
df_test.to_csv('./dacon/data/df_test.csv', sep = ',')
x_train = x_train.to_numpy()                                               #데이터 프레임을 넘파이 파일로 변환
sun_pred = x_pred.to_numpy()                                                #데이터 프레임을 넘파이 파일로 변환1

print(x_train.shape) #(52560, 9)
print(sun_pred.shape) #(3888, 9)
x_train = x_train.reshape(1095, 48, 9)
sub_test = sun_pred.reshape(3888, 1, 9)


# 1.2 비교 데이터 추출
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

x, y = split_xy(x_train, 7, 2)                      #x행 수와 y(비교행 수)
print(x.shape)                                      #(1087, 7, 48, 9)
print(y.shape)                                      #(1087, 2, 48, 9) ...3넣어서 3일치 나올뻔 ㅜㅜ
#print(x_pred.shape)                                 #

#train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 50)


#MinMaxScaler_reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]*y_train.shape[3])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]*y_test.shape[3])
# sub_test = sub_test.reshape(sub_test.shape[0], sub_test.shape[1]*sub_test.shape[2])

#MinMaxScaler 1, 2, 3
scaler1 = MinMaxScaler()
scaler1.fit(x_train)
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)

scaler2 = MinMaxScaler()
scaler2.fit(y_train)
y_train = scaler2.transform(y_train)
y_test = scaler2.transform(y_test)

# scaler3 = MinMaxScaler()
# scaler3.fit(sub_test)
# sub_test = scaler3.transform(sub_test)

print(x_train.shape)                                    #(869, 3024)
print(x_test.shape)                                     #(218, 3024)
print(y_train.shape)                                    #(869, 864)
print(y_test.shape)                                     #(218, 864)
# print(sub_test.shape)                                     #(81, 432)


#LSTM reshape
x_train = x_train.reshape(869, 7, 48, 9)                      # (869, 7, 48, 6)
x_test = x_test.reshape(218, 7, 48, 9)                       # (218, 7, 48, 6)
y_train = y_train.reshape(869, 2, 48, 9)                      # (869, 2, 48, 6)
y_test = y_test.reshape(218, 2, 48, 9)                       # (218, 2, 48, 6)
# sub_test = sub_test.reshape(81, 48, 9)
print(x_train.shape)    
print(x_test.shape)                                 
print(y_train.shape)                                
print(y_test.shape)                                 


#LSTM reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2], x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2], x_test.shape[3])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2], y_train.shape[3])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2], y_test.shape[3])


#2. 모델링
input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
lstm = LSTM(10, activation='relu', input_shape = (x_train.shape[1], x_train.shape[2]))(input1)
dense1 = Dense(16, activation='relu')(lstm)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
outputs = Dense(9)(dense1)
# outputs = Reshape((9))(dense1)
#모델 선언
model = Model(inputs = input1, outputs = outputs)
model.summary()


#3컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './dacon/data/MCP/sun_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
red_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
es = EarlyStopping(monitor='loss', patience=25, mode = 'auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=3, batch_size=10, validation_split = 0.2, verbose=1, callbacks=[es, cp, red_lr])


#4. 평가. 예측
loss, mae = model.evaluate(x_test, y_test)
y_pred = model.predict(y_test)
print('loss : ', loss)
print('mae : ', mae)
print(y_pred)
print(y_pred.shape) #(81, 48, 9)

# y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2])

# y_pred.to_csv('./dacon/data/sample_submission_test.csv', header = False, index = False)
# np.savetxt("./dacon/data/sample_submission.csv", y_pred, delimiter=",")
# y_pred.tofile('./dacon/data/sample_submission.csv', sep = ',')
# y_pred.to_csv("./dacon/data/sample_submission.csv",index=False)

'''
# submission
sub = pd.read_csv('./dacon/data/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred[:,0]
'''