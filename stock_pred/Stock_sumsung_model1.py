import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#0. 함수정의
def split_x(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size)]  
        aaa.append(subset) 
    print(type(aaa)) 
    return np.array(aaa) 
a = np.array(range(1, 11)) # a에 0~10까지 집어 넣는다
size = 4 # 데이터를 5개씩 끊어서 리스트 구성 (5개씩 끊어서) 5개 열로 구성함
b = np.array(range(1, 11)) # a에 0~10까지 집어 넣는다
dataset = split_x(a, size) 
print(dataset) #(1~100)
x = dataset[:, :5]
y = dataset[:, -1]
dataset_pred = np.array(split_x(b, 6))  
x_pred = dataset_pred[:, :5]
y_pred = dataset_pred[ :,-1]
print(x_pred)
print(y_pred)

#1.데이터
datasets = pd.read_csv('../data/csv/삼성전자.csv', index_col = 0 , header=0, encoding='CP949', thousands=',')
datasets.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#  0         1       2     3         4         5         6         7        8        9         10           11                 12           13                     
#['open', 'high', 'low', 'closing', 'rate', 'volume', 'amount', 'credit', 'indiv', 'inst', 'foreign', 'foreigners(su)', 'programs', 'foreigners(%)']

#1.1 전처리_결측치 제거, 미사용 컬럼제거
datasets_1 = datasets.iloc[:662,:]
datasets_2 = datasets.iloc[665:,:]
datasets = pd.concat([datasets_1,datasets_2]) #from.iwillbeaprogramer
datasets.drop([5,6,7,12,13], axis='columns', inplace=True)
#1.2 데이터 변환
datasets.iloc[662:,0:4] = datasets.iloc[662:,0:4]/50.0
datasets.iloc[662:,5] = datasets.iloc[662:,5]*50
datasets = datasets.astype('float32')
np.save("../data/npy/삼성전자.npy",arr=datasets)

data_x = datasets.iloc[:, :]
data_y = datasets.iloc[:,[3]]
print(data_x.shape, data_y.shape) #(2397, 9) (2397, 1)

x_data = data_x.to_numpy()
y_data = data_y.to_numpy()
print(x_data)
print(y_data)
print(x_data.shape)# (2397, 9)
print(y_data.shape)# (2397, 1)

x = x_data
y = y_data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state = 60)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle= True, random_state = 60)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

#2. 모델링
# input1 = Input(shape=(x.shape[1],1))
# lstm = LSTM(300, activation='relu')(input1)
# drop1 = Dropout(0.2)(lstm)
# dense1 = Dense(172, activation='relu')(drop1)
# drop2 = Dropout(0.2)(dense1)
# dense1 = Dense(152, activation='relu')(drop2)
# drop3 = Dropout(0.21)(dense1)
# dense1 = Dense(152, activation='relu')(drop3) 
# dense1 = Dense(150, activation='relu')(dense1)
# dense1 = Dense(132, activation='relu')(dense1)
# dense1 = Dense(48, activation='relu')(dense1)
# outputs = Dense(1, activation= 'relu')(dense1)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

# #3.컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelCheckpoint/Stock_SSD_14_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
# early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
# hist = model.fit(x_train, y_train, epochs = 10000, batch_size = 32, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

model = load_model('../data/h5/Stock_SSD_14_model2.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test[-5:-1])
print(y_predict[-1])
# print(y_train[-5:-1])

# # 시각화
# import matplotlib.pyplot as plt
# plt.rc('font', family='Malgun Gothic')
# plt.figure(figsize=(10,6))
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
# plt.grid()
# plt.title('cost_loss') #loss, cost
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.subplot(2,1,2)
# plt.plot(hist.history['mae'], marker = '.', c = 'red', label = 'mae')
# plt.plot(hist.history['val_mae'], marker = '.', c = 'blue', label = 'val_mae')
# plt.grid()
# plt.title('cost_mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.show()