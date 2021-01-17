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

#2. 모델링
#3. 컴파일, 훈련
model = load_model('./stock_pred/Stock_SSD_3_model2.h5')#모델로드
# model.load_weights('./stock_pred/Stock_SSD3_weight.h5') #weight로드


#4. 평가, 예측
loss, mae = model.evaluate([x1_test, x2_test], y_test ,batch_size=7)
y_predict = model.predict([x1_test, x2_test])
x_predict = model.predict([x1_pred, x2_pred])
print('loss : ', loss)
print('mae : ', mae)
y_predict = model.predict([x1_test, x2_test])
x_predict = model.predict([x1_pred,x2_pred])
print('18일, 19일 시가 :', x_predict[0])
print('19일 시가 :', x_predict[0,1:])

# loss :  333201.21875
# mae :  424.3497619628906
# 18일 시가 : [87843.32 87613.21]
# 19일 시가 : [87613.21]
