#보스턴 집값
#실습:모델구성
import numpy as np
x_data = np.load('../data/npy/boston_x.npy')
y_data = np.load('../data/npy/boston_y.npy')

#1. 데이터
x = x_data
y = y_data
print(x.shape) #(506, 13)
print(y.shape) #(506,   )

# 1_2. 데이터 전처리(MinMaxScalar)
# ex 0~711 = 최댓값으로 나눈다  0~711/711
# X - 최소값 / 최대값 - 최소값
# print("===================")
# print(x[:5]) # 0~4
# print(y[:10]) 
# print(np.max(x), np.min(x)) # max값 min값
# print(dataset.feature_names)
#print(dataset.DESCR) #묘사

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
#(323, 13) (323,) (102, 13) (102,) (81, 13) (81,)
#2. 모델링
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
input1 = Input(shape=(13,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(60, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(16, activation='relu')(dense1) 
dense1 = Dense(7, activation='relu')(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()
# model.save('../data/h5/k52_4_model1.h5')

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=20, mode = 'auto')
# modelpath = '../data/modelCheckpoint/k52_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath= modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
# tb = TensorBoard(log_dir='./graph', histogram_freq = 0, write_graph=True, write_images=True) #그림출력 /graph폴더에 저장
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=6, validation_data= (x_val, y_val), callbacks = [early_stopping])#, cp, tb])
# model.save('../data/h5/k52_4_model2.h5')
# model.save_weights('../data/h5/k52_4_weight.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) 

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )


# 시각화
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['axes.unicode_minus'] = False 
# matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(10,10))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('손실비용') #loss, cost
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2)
plt.plot(hist.history['mae'], marker = '.', c = 'red', label = 'mae')
plt.plot(hist.history['val_mae'], marker = '.', c = 'blue', label = 'val_mae')
plt.grid()

plt.title('정확도')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()


#전처리 전
# loss :  16.55705451965332
# mae :  3.3871774673461914
# RMSE : 4.069036165639308
# R2 :  0.8019086688524137

#통째로 전처리
# loss :  11.465134620666504
# mae :  2.5706095695495605
# RMSE : 3.386020620416784
# R2 :  0.8628292327610475

#제대로 전처리(?)
# loss :  531.5300903320312
# mae :  21.24960708618164
# RMSE : 23.054936080104717
# R2 :  -5.359313211830821

#발리데이션 test분리
# loss :  5.44482421875
# mae :  1.7919334173202515
# RMSE : 2.3334145056348183
# R2 :  0.9430991642272919

# loss :  10.888834953308105
# mae :  2.4144272804260254
# RMSE : 3.299823642181916
# R2 :  0.8697241755661469

# loss :  7.084146022796631 MinMax처리
# mae :  1.9379582405090332
# RMSE : 2.6616056446549266
# R2 :  0.9152441295579581

"""
#전처리가 된 데이터(정규화)
#[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00] = 되어있지않음...

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
"""