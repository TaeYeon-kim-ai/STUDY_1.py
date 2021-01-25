#데이터 중복검사 .duplicateds      .duplicateds.sum()

#자르기 - 2일 골라내기 - 예측 /// 7일될때마다 2개씩 골라내기

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

#1. 데이터
# print(df_train.shape) #(52464, 9)
# print(x_pred.shape) #(3888, 7)
x_train = df_train

#numpy 변환
x_train = preprocess_data(df_train) #뒤에 두개 제외 Y값
x_train = df_train.to_numpy()
x_pred = x_pred.to_numpy()

#데이터 분할
def split_xy(dataset, time_steps):
    x, y1, y2 = [], [], []
    for i in range(len(dataset)):
        x_data_end = i + time_steps
        if x_data_end > len(dataset) -1:
            break
        tmp_x = dataset[i:x_data_end, :-2]
        tmp_y1 = dataset[x_data_end-1:x_data_end,-1]
        tmp_y2 = dataset[x_data_end-1:x_data_end, -2]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

x, y1, y2 = split_xy(x_train, 1)
print(x.shape) # (52463, 1, 7)
print(y1.shape) #(52463, 1)
print(y2.shape)# (52463, 1)
x_pred = x_pred.reshape(x_pred.shape[0], 1, 7)
print(x_pred.shape) #(3888, 7)


#train_t_s
x_train, x_test, y1_train, y1_test, y2_train, y2_test  = train_test_split(x, y1, y2, train_size = 0.7, shuffle=True, random_state = 0)
x_train, x_val, y1_train, y1_val, y2_train, y2_val  = train_test_split(x_train, y1_train, y2_train, train_size = 0.7, shuffle=True, random_state = 0)

print(x_train.shape) #(25706, 1, 7)
print(x_test.shape) #(15739, 1, 7)
print(x_val.shape) #(11018, 1, 7)

x_train = x_train.reshape(x_train.shape[0], 7)
x_test = x_test.reshape(x_test.shape[0], 7)
x_val = x_val.reshape(x_val.shape[0], 7)

#StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], 1, 7)
x_test = x_test.reshape(x_test.shape[0], 1, 7)
x_val = x_val.reshape(x_val.shape[0], 1, 7)

#quantile_loss
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# def quantile_loss(y_true, y_pred) : 
#     qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
#     q = tf.constant(np.array([qs]), dtype = tf.float32) #qs를 텐서플로의 (컨스턴트형식) 상수(바뀌지 않는값)형태로 바꾸겠다. tf 1.x
#                                                         #tf.constant 검색                
#     e = y_true = y_pred #y_true(실제값)에서 y_pred(예측값)
#     v = tf.maximum(q*e, (q-1)*e) 
#     return K.mean(v)

#2. 모델링
def conv1d_model() :
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2]))
    conv1d = Conv1D(256, 2, activation= 'relu', padding= 'SAME',input_shape = (x_train.shape[1], x_train.shape[2]))(inputs)
    drop = Dropout(0.1)(conv1d)
    conv1d = Conv1D(256, 2, padding= 'SAME', activation='relu')(drop)
    conv1d = Conv1D(104, 2, padding= 'SAME', activation='relu')(conv1d)
    conv1d = Conv1D(88, 2, padding= 'SAME', activation='relu')(conv1d)
    flt = Flatten()(conv1d)
    dense1 = Dense(72, activation='relu')(flt)
    dense1 = Dense(64, activation='relu')(dense1)
    dense1 = Dense(64, activation='relu')(dense1)
    dense1 = Dense(64, activation='relu')(dense1)
    dense1 = Dense(32, activation='relu')(dense1)
    dense1 = Dense(16, activation='relu')(dense1)
    outputs = Dense(1)(dense1)
    model = Model(inputs = inputs, outputs = outputs)
    return model

#3. 컴파일, 훈련
#optimizer

# optimizer = Adam(lr = 0.1)
optimizer = Adam(lr = 0.01)
# optimizer = Adam(lr = 0.001)
# optimizer = Adam(lr = 0.0001)
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
lr = ReduceLROnPlateau(monitor= 'val_loss', patience = 3, factor= 0.5)
ep = 100
bs = 32

x = []
for i in quantiles:
    model = conv1d_model() 
    # modelpath = './dacon/data/MCP/dacon_01_y1_{epoch:02d}-{val_loss:.4f}.hdf5'
    # cp = ModelCheckpoint(modelpath,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = [lambda y_true, y_pred: quantile_loss(i, y_true, y_pred)], optimizer = optimizer , metrics = [lambda y, y_pred: quantile_loss(i, y, y_pred)])
    model.fit(x_train, y1_train, epochs = ep , batch_size = bs, validation_data= (x_val, y1_val), verbose = 1, callbacks = [es, lr]) 
    pred = pd.DataFrame(model.predict(x_pred).round(2)) #dacon LGBM
    x.append(pred)
df_tmp1 = pd.concat(x, axis= 1)
df_tmp1[df_tmp1<0] = 0
np_tmp1 = df_tmp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = np_tmp1

x = []
for i in quantiles:
    model = conv1d_model() 
    # modelpath = './dacon/data/MCP/dacon_01_y2_{epoch:02d}-{val_loss:.4f}.hdf5'
    # cp = ModelCheckpoint(modelpath,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = [lambda y_true, y_pred: quantile_loss(i, y_true, y_pred)], optimizer = optimizer , metrics = [lambda y, y_pred: quantile_loss(i, y, y_pred)])
    model.fit(x_train, y2_train, epochs = ep , batch_size = bs, validation_data= (x_val, y2_val), verbose = 1, callbacks = [es, lr]) 
    pred = pd.DataFrame(model.predict(x_pred).round(2)) #dacon LGBM
    x.append(pred)
df_tmp2 = pd.concat(x, axis= 1)
df_tmp2[df_tmp2<0] = 0
np_tmp2 = df_tmp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = np_tmp2

#저장
submission.to_csv('./dacon/data/submission_210125_1.csv', index=False)