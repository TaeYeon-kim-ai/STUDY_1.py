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

#GHI추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    return data

def preprocess_data(data, is_train=True):
    data = Add_features(data) #GHI추가
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']] # 컬럼 추출

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # -48시간을 열1개 추가하여 채움 +1일 타겟 데이터
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') #  -48시간을 열2개 추가하여 채움  +2일 타겟 데이터
        temp = temp.dropna() #결측값 제거
        
        return temp.iloc[:-96] #예측날짜 제거 (-2일)

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']] # 컬럼 추출
                               
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

#numpy 변환
x_pred = pd.concat(df_test)
x_pred = x_pred.to_numpy()
x_train = df_train.to_numpy()

print(df_train.shape) #(52464, 10)
print(x_pred.shape) #(3888, 8)
#============================================

# #MinMax
# scaler = MinMaxScaler()
# scaler.fit(x_train[:,:-2])
# x_train[:,:-2] = scaler.transform(x_train[:,:-2])
# x_pred = scaler.transform(x_pred)

scaler = StandardScaler() 
scaler.fit(x_train[:,:-2])
x_train[:,:-2] = scaler.transform(x_train[:,:-2])
x_pred = scaler.transform(x_pred)

#데이터 분할
def split_xy(dataset, time_steps):
    x, y1, y2 = [], [], []
    for i in range(len(dataset)):
        x_data_end = i + time_steps
        if x_data_end > len(dataset) -1:
            break
        tmp_x = dataset[i:x_data_end, :-2]
        tmp_y1 = dataset[x_data_end-1:x_data_end,-1]
        tmp_y2 = dataset[x_data_end-1:x_data_end,-2]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

x, y1, y2 = split_xy(x_train, 1)

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

x_pred = split_x(x_pred,1)
print(x.shape) #(52463, 1, 7)
print(y1.shape) #(52367, 1)
print(y2.shape) #(52367, 1)
print(x_pred.shape) #(3888, 1, 7)

#      Hour  TARGET  DHI  DNI   WS     RH     T
# 288     0     0.0    0    0  0.8  80.92  -2.8

#============================================

#train_sprit_t
x_train, x_test, y1_train, y1_test = train_test_split(x, y1, train_size = 0.7, shuffle = True, random_state = 0)
x_train, x_val, y1_train, y1_val = train_test_split(x_train, y1_train, train_size = 0.7, shuffle = True, random_state = 0)
x_train, x_test, y2_train, y2_test = train_test_split(x, y2, train_size = 0.7, shuffle = True, random_state = 0)
x_train, x_val, y2_train, y2_val = train_test_split(x_train, y2_train, train_size = 0.7, shuffle = True, random_state = 0)


#2. 모델링
def co1_model() :
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2]))
    conv1d = Conv1D(256, 2, activation= 'relu', padding= 'SAME',input_shape = (x_train.shape[1], x_train.shape[2]))(inputs)
    conv1d = Conv1D(256, 2, padding= 'SAME', activation='relu')(conv1d)
    conv1d = Conv1D(128, 2, padding= 'SAME', activation='relu')(conv1d)
    conv1d = Conv1D(128, 2, padding= 'SAME', activation='relu')(conv1d)
    flt = Flatten()(conv1d)
    dense1 = Dense(32, activation='relu')(flt)
    dense1 = Dense(32, activation='relu')(dense1)
    dense1 = Dense(16, activation='relu')(dense1)
    outputs = Dense(1)(dense1)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    return model

#3. 컴파일, 훈련
#loss 함수
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]   

ep = 10
bts = 64
es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
lr = ReduceLROnPlateau(monitor= 'val_loss', patience = 3, factor= 0.3,  verbose = 1)
optimizer = Adam(lr = 0.01)

x = []
for j in q:
    model = co1_model()
    modelpath = './dacon/data/MCP/dacon_01_y1_{epoch:02d}-{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(modelpath,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = [lambda y_true, y_pred: quantile_loss(j, y_true, y_pred)], optimizer = optimizer , metrics = ['mae'])
    model.fit(x_train, y1_train, epochs = ep , batch_size = bts, validation_data= (x_val, y1_val), verbose = 1, callbacks = [es, cp ,lr]) 
    #저장
    pred = pd.DataFrame(model.predict(x_pred).round(2))
    x.append(pred)
df_temp1 = pd.concat(x, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

x = []
for j in q:
    model = co1_model()
    modelpath = './dacon/data/MCP/dacon_01_y1_{epoch:02d}-{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(modelpath,save_best_only=True,monitor = 'val_loss')
    model.compile(loss = [lambda y_true, y_pred: quantile_loss(j, y_true, y_pred)], optimizer = optimizer , metrics = ['mae'])
    model.fit(x_train, y2_train, epochs = ep , batch_size = bts, validation_data= (x_val, y2_val), verbose = 1, callbacks = [es, cp ,lr]) 
    #저장
    pred = pd.DataFrame(model.predict(x_pred).round(2))
    x.append(pred)
df_temp2 = pd.concat(x, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('./dacon/data/submission_210126_sub1.csv', index=False)

#평가, 예측
loss, mae = model.evaluate(x_test, [y1_test, y2_test],batch_size=7)
y_predict = model.predict(x_test)
print('loss : ', loss)
print('mae : ', mae)

# ep 32 / bts 64
# loss :  1.2249244451522827
# mae :  8.092819213867188

#ep 32 / bts 32
# loss :  1.355821132659912
# mae :  8.446158409118652
