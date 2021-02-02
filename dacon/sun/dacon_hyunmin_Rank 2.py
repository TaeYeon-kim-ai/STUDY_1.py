# 상관계수 높은 열만 사용하기
# 7일치로 2일 예측하기
# epochs 10으로 했는데도 점수 잘 나온다

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# 만들고 싶은 모양 : 같은 시간대 별로 묶는다.
# 6일치 데이터로 2일치를 예측한다.
# print(x.shape)     # (48, N, 5, 8)
# print(y.shape)     # (48, N, 1, 2)

# print(x_pred.shape)  # (48, 81, 5, 8)
# y_pred.shape (48, 81, 1, 2)

##############################################################

# 파일 불러오기
train = pd.read_csv('./dacon/data/train/train.csv')
# print(train.shape)  # (52560, 9)
# print("df_train null : ", train.duplicated().sum())   # 0

submission = pd.read_csv('./data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)

##############################################################

#1. DATA

# 함수 : GHI column 추가
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

# 함수 : train data column 정리
# 끝에 다음날, 다다음날 TARGET 데이터 column을 추가한다.
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    # print(data.columns) 
    # Index(['Day', 'T-Td', 'Td', 'GHI', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH','T', 'TARGET'], dtype='object')
    temp = data.copy()
    temp = temp[['Day','TARGET','GHI','DHI','DNI','T-Td']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96, :]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:     
        return temp.iloc[-48*6:, 1:]  # 6일치만 사용

#함수 : 같은 시간대끼리 모으기
def same_train(train) :
    temp = train.copy()
    x = list()
    final_x = list()
    for i in range(48) :
        same_time = pd.DataFrame()
        for j in range(int(len(temp)/48)) :
            tmp = temp.iloc[i + 48*j, : ]
            tmp = tmp.to_numpy()
            tmp = tmp.reshape(1, tmp.shape[0])
            tmp = pd.DataFrame(tmp)
            # print(tmp)
            same_time = pd.concat([same_time, tmp])
        x = same_time.to_numpy()
        final_x.append(x)
    return np.array(final_x)

# print(len(train)) # 52560
# print(same_train(train).shape) # (48, 1095, 9)

# 함수 : 시계열 데이터로 자르기 (x는 6행씩, y는 1행씩)
def split_xy(dataset, time_steps) :  # data, 6
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        y_end = x_end-1
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i : x_end, 1:-2]      # ['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td']
        tmp_y = dataset[y_end, -2:]           # ['Target1', 'Target2']
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

##############################################################

# train data
df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 8)
# print(df_train.columns) 
# Index(['Day', 'TARGET', 'GHI', 'DHI', 'DNI', 'T-Td', 'Target1', 'Target2'], dtype='object')


# 상관계수 확인
# print(df_train.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_train.corr(),square=True, annot=True, cbar=True)
# plt.show()
# > 기준 : Target1, Target2
# > 상관계수 0.6 이상 : Target, GHI, DHI, DNI, T-Td

# 같은 시간대 별로 묶기
same_time = same_train(df_train)
# print(same_time.shape)  # (48, 1093, 8)
# print(same_time[0:3, :5 :])

# X = same_time.to_numpy()
# print(X.shape)      # (52464, 8)

x, y = list(), list()
for i in range(48):
    tmp1,tmp2 = split_xy(same_time[i], 6)
    x.append(tmp1)
    y.append(tmp2)

x = np.array(x)
y = np.array(y)
print("x.shape : ", x.shape) # (48, 1088, 6, 5)
print("y.shape : ", y.shape) # (48, 1088, 2)

y = y.reshape(48, 1088, 1, 2)


# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td'], dtype='object')
    temp = pd.DataFrame(temp)
    temp = same_train(temp)
    df_test.append(temp)

x_pred = np.array(df_test)
print("x_pred.shape : ", x_pred.shape) # (81, 48, 6, 5)


##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=332)
x_train, x_val, y_train, y_val, = \
    train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=332)

# print(x_train.shape)    # (30, 1088, 6, 5)
# print(x_test.shape)     # (10, 1088, 6, 5)
# print(x_val.shape)      # (8, 1088, 6, 5)

# print(y_train.shape)   # (30, 1088, 1, 2)
# print(y_test.shape)    # (10, 1088, 1, 2)
# print(y_val.shape)     # (8, 1088, 1, 2)

# StandardScaler를 하기 위해서 2차원으로 변환
x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1] * x_train.shape[2], x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1] * x_test.shape[2], x_test.shape[3])
x_val = x_val.reshape(x_val.shape[0] * x_val.shape[1] * x_val.shape[2], x_val.shape[3])
x_pred = x_pred.reshape(x_pred.shape[0] * x_pred.shape[1] * x_pred.shape[2], x_pred.shape[3])

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(30 * 1088, 6, 5)
x_test = x_test.reshape(10 * 1088, 6, 5)
x_val = x_val.reshape(8 * 1088, 6, 5)
x_pred = x_pred.reshape(81 * 48, 6, 5)

y_train = y_train.reshape(30 * 1088, 1, 2)
y_test = y_test.reshape(10 * 1088, 1, 2)
y_val = y_val.reshape(8 * 1088, 1, 2)

##############################################################

#2. Modeling
#3. Compile, Train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)


#2. Modeling
def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 5, 6)
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Reshape((1, 2)))  # output (N, 1, 2)
    model.add(Dense(2))
    return model

##############################################################

loss_list = list()

# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles = [0.2, 0.4, 0.6, 0.7, 0.9]
quantiles = [0.7, 0.9]


epoch = 50
batch = 8

for q in quantiles :
    print(f"\n>>>>>>>>>>>>>>>>>>>>>>  modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    #2. Modeling
    model = modeling()
    # model.summary()

    #3. Compile, Train
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')
    
    cp_save = f'../data/modelcheckpoint/solar_0126_s8_q_{q:.1f}.hdf5'
    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, verbose=1)

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

    # 4. Evaluate, Predict
    loss = model.evaluate(x_test, y_test,batch_size=8)
    print("loss : ", loss)
    loss_list.append(loss)  # loss 기록

    y_pred = model.predict(x_pred)
    # print("1 ",y_pred.shape)    # (3888, 1, 2)
    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])) # (3888, 2)
    # print("2 ",y_pred.shape)    #(3888, 2)
    y_pred = pd.concat([y_pred], axis=1)
    y_pred[y_pred<0] = 0
    y_pred = y_pred.to_numpy()

    # submission
    column_name = f'q_{q}'
    submission.loc[submission.id.str.contains("Day7"), column_name] = np.around(y_pred[:, 0],3)   # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = np.around(y_pred[:, 1],3)   # Day8 (3888, 9)
    submission.to_csv(f'../data/DACON_0126/submission_0126_8_{q}.csv', index = False)  # score : 

loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
print("loss_mean : ", loss_mean)    # 
print("loss_list : ", loss_list )


# to csv
submission.to_csv('../data/DACON_0126/submission_0126_8.csv', index=False)  # score : 