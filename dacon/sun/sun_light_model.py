import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
'''
# train.csv : 훈련용 데이터 (1개 파일)
# - 3년(Day 0~ Day1094) 동안의 기상 데이터, 발전량(TARGET) 데이터  
# test.csv : 정답용 데이터 (81개 파일)
# - 2년 동안의 기상 데이터, 발전량(TARGET) 데이터 제공 
# - 각 파일(*.csv)은 7일(Day 0~ Day6) 동안의 기상 데이터, 발전량(TARGET) 데이터로 구성
# - 파일명 예시: 0.csv, 1.csv, 2.csv, …, 79.csv, 80.csv (순서는 랜덤이므로, 시계열 순서와 무관)
# - 각 파일의 7일(Day 0~ Day6) 동안의 데이터 전체 혹은 일부를 인풋으로 사용하여, 향후 2일(Day7 ~ Day8) 동안의 30분 간격의 발전량(TARGET)을 예측 (1일당 48개씩 총 96개 타임스텝에 대한 예측)
# sample_submission.csv : 정답제출 파일
# - test 폴더의 각 파일에 대하여, 시간대별 발전량을 9개의 Quantile(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)에 맞춰 예측
# - “파일명_날짜_시간” 형식(예시: 0.csv_Day7_0h00m ⇒ 0.csv 파일의 7일차 0시00분 예측 값)에 유의
# Hour - 시간
# Minute - 분
# DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
# DNI - 직달일사량(Direct Normal Irradiance (W/m2))
# WS - 풍속(Wind Speed (m/s))
# RH - 상대습도(Relative Humidity (%))
# T - 기온(Temperature (Degree C))
# Target - 태양광 발전량 (kW)
'''
#1.데이터
data = np.load('./dacon1/data/train/train.csv', index_col=0, header=0, encoding='CP949')
test = np.load('./dacon1/data/test/test_merge.csv', index_col=0, header=0, encoding='CP949')

#1.1 비교 데이터 추출

#1.2 함수정의
def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
print(df_train.shape) #(52464, 9)
print(test.shape) #(27216, 8)

'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, randam_state = 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, randam_state = 100)


#2. 모델링
inputs = Input(shape = (x.shape[1], x.shape[2], x.shape[3]))
lstm = LSTM(250, activation='relu', input_shape = (x.shape[1], x.shape[2], x.shape[3]))
dense1 = Dense(64, activation='relu')(lstm)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
outputs = Dense(2, activation='relu')(dense1)

#모델 선언
model = Model(input = inputs, outputs = outputs)

#3컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './dacon1/data/MCP/sun_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
es = EarlyStopping(monitor='loss', patience=25, mode = 'auto')
model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data = (x_val, y_val), verbose=1, callbacks=[es, cp, reduce_lr])
#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=7)
y_predict = model.predict(x_test)
x_predict = model.predict(y)
print('loss : ', loss)
print('mae : ', mae)
print('day7,8', x_predict)

#RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)
'''