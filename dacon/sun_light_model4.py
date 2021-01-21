#자르기 - 2일 골라내기 - 예측 /// 7일될때마다 2개씩 골라내기

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
train.iloc[48:96] #기존 train 48~95까지 자르기
train.iloc[48+48:96+48] #train data 96 ~ +48까지 잘라서 보기

#============================================
#test데이터 병합
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv' #test파일 불러오기
    temp = pd.read_csv(file_path) #pandas로 파일 읽기
    temp = preprocess_data(temp, is_train=False) 
    df_test.append(temp)

x_pred = pd.concat(df_test)
#test병합 끝.
#============================================

#1. 데이터
print(df_train.shape) #(52464, 9)
print(x_pred.shape) #(3888, 7)

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], train_size = 0.7, random_state = 0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], train_size = 0.7, random_state = 0)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train1, y_train1, train_size = 0.7, random_state = 0)
x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train2, y_train2, train_size = 0.7, random_state = 0)
#1.1 데이터 전처리
print(x_train1.shape)#(25706, 7)
print(x_train2.shape)#(25706, 7)
print(x_test1.shape)#(15740, 7)
print(x_test2.shape)#(15740, 7)
print(x_val1.shape)#(11018, 7)
print(x_val2.shape)#(11018, 7)
'''
#1.1데이터 전처리
#StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_train1)
x_train1 = scaler1.transform(x_train1)
x_test1 = scaler1.transform(x_test1)
x_val1 = scaler1.transform(x_val1)

scaler2 = StandardScaler()
scaler2.fit(x_train2)
x_train2 = scaler1.transform(x_train2)
x_test2 = scaler1.transform(x_test2)
x_val2 = scaler1.transform(x_val2)

#1.2. 데이터 reshape

#2. 모델링
def Seq(q, x_train, y_train, x_valid, y_valid, x_pred): #q값 
    
    # (a) Modeling  
    model = input1 = Input(shape()
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2)) 
                #시리즈를 주어진 소수 자릿수로 반올림함 .00
    return pred, model


#3. 컴파일, 훈련




'''