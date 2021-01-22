import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#0. 함수정의
train = pd.read_csv('./dacon/data/train/train.csv')
submission = pd.read_csv('./dacon/data/sample_submission.csv')


def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']] # 컬럼 추출

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # -48시간을 열1개 추가하여 채움 +1일 타겟 데이터
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') #  -48시간을 열2개 추가하여 채움  +2일 타겟 데이터
        temp = temp.dropna() #결측값 제거
        
        return temp.iloc[:-96] #예측날짜 제거 (-2일)

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']] 
                               
        return temp.iloc[-48:, :] # if문 진행 -48~끝까지, 뒤에서 6일째 데이터만 남김 - if 데이터 +1~2일 예측 ex_ 1일 - 2, 3일 예측 / 2일 - 3,4일 예측

df_train = preprocess_data(train)
df_train.iloc[:48] # T1, T2추가된 df_train 0~48까지 자르기

#============================================
#test데이터 병합
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv' #test파일 불러오기
    temp = pd.read_csv(file_path) #pandas로 파일 읽기
    temp = preprocess_data(temp, is_train=False) 
    df_test.append(temp)

x_pred = pd.concat(df_test)
#============================================

#numpy 변환
x_train = preprocess_data(df_train) #뒤에 두개 제외 Y값
x_train = x_train.to_numpy()
x_pred = x_pred.to_numpy()

#데이터 분할
def split_xy(dataset, time_steps):
    x, y = [], []
    for i in range(len(dataset)):
        x_data_end = i + time_steps
        if x_data_end > len(dataset) -1:
            break
        tmp_x = dataset[i:x_data_end, :-2]
        tmp_y = dataset[x_data_end-1:x_data_end,-2:]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(x_train, 1)

print(x.shape) #(52463, 1, 7)
print(y.shape) #(52463, 1, 2)
print(x_pred.shape) #(3888, 7)
print(x) 
print(y)
print(x_pred) 

#      Hour  TARGET  DHI  DNI   WS     RH     T
# 288     0     0.0    0    0  0.8  80.92  -2.8

#============================================

#train_sprit_t
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False, random_state = 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False, random_state = 0)

# #변환
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

print(x_train.shape)#(33514, 7)
print(x_test.shape)#(10474, 7)
print(x_val.shape)#(8379, 7)
print(y_train.shape)#(33514, 7)
print(y_test.shape)#(10474, 7)
print(y_val.shape)#(8379, 7)

# #shane
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], 7, 1)
x_test = x_test.reshape(x_test.shape[0], 7, 1)
x_val = x_val.reshape(x_val.shape[0], 7, 1)

y_train = y_train.reshape(y_train.shape[0], 2, 1)
y_test = y_test.reshape(y_test.shape[0], 2, 1)
y_val = y_val.reshape(y_val.shape[0], 2, 1)

# print(x_pred)

#2. 모델링
inputs = Input(shape = (x_train.shape[1], x_train.shape[2]))
conv1d = Conv1D(128, 2, activation= 'relu', padding= 'SAME',input_shape = (x_train.shape[1], x_train.shape[2]))(inputs)
drop = Dropout(0.1)(conv1d)
conv1d = Conv1D(256, 2, padding= 'SAME', activation='relu')(drop)
conv1d = Conv1D(128, 2, padding= 'SAME', activation='relu')(conv1d)
flt = Flatten()(conv1d)
dense1 = Dense(64, activation='relu')(flt)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)
outputs = Dense(2)(dense1)
model = Model(inputs = inputs, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
#loss 함수
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
lr = ReduceLROnPlateau(monitor= 'val_loss', patience = 3, factor= 0.5)
optimizer = Adam(lr = 0.001)

for j in q:
    # modelpath = './dacon/data/MCP/dacon_01_y1_{epoch:02d}-{val_loss:.4f}.hdf5'
    # cp = ModelCheckpoint(modelpath,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = [lambda y_true, y_pred: quantile_loss(j, y_true, y_pred)], optimizer = optimizer , metrics = ['mae'])
    model.fit(x_train, y_train, epochs = 100 , batch_size = 32, validation_data= (x_val, y_val), verbose = 1, callbacks = [es, lr]) 
    #저장
    temp = model.predict(x_pred).round(2)
    print(temp, temp.shape)
    col = 'q_' + str(j)
    submission.loc[submission.id.str.contains("Day7"),col] = temp[:,0]
    submission.loc[submission.id.str.contains("Day8"),col] = temp[:,1]

submission.to_csv('./dacon/data/submission_210121_7.csv', index=False)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test ,batch_size=7)
y_predict = model.predict(x_test)
pred = pd.DataFrame(model.predict(x_pred).round(2)) #dacon LGBM
print('loss : ', loss)
print('mae : ', mae)
print('predict : ', pred)