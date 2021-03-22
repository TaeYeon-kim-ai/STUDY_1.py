#실습
#모델을 만들어라

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#1. 데이터(기본)
#npy 불러오기
x_train = np.load('../data/image/brain/npy/keras66_train_x.npy') #(160, 150, 150, 3)
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy') #(120, 150, 150, 3)
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy') #(160,)
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy') # (120,)

print(x_train.shape, y_train.shape) #(160, 150, 150, 3)
print(x_test.shape, y_test.shape) # (120, 150, 150, 3)


#1.1 전처리 / minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)


#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten
input1 = Input(shape=(x_train.shape[1], x_train.shape[2] ,x_train.shape[3]))
x = Conv2D(64, 2, activation='relu')(input1)
x = Conv2D(128, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x) 
x = BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(40, activation='relu')(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 30, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 32 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# loss :  0.6146066784858704
# acc :  0.7416666746139526

# y_predict = model.predict(x_test[:10])
# # y_pred = list(map(int,np.round(y_predict,0)))
# result = np.transpose(y_predict)
# print(np.argmax(result[:5], axis=-1))
# print(np.argmax(y_test[:5], axis=-1))



#시각화
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(fihsize = (10,6))

# plt.subplot(2,1,1) #2행 1열 중 첫번때
# plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
# plt.grid()

# plt.title('cost_loss') #loss,cost #타이틀깨진것 한글처리 해둘 것
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.subplot(2,1,2) #2행 2열중 2번째
# plt.plot(hist.history['acc'], marker = '.', c='red')
# plt.plot(hist.history['val_acc'], marker = '.', c='blue')
# plt.grid() #그래프 격자(모눈종이 형태)

# plt.title('cost_acc')  #타이틀깨진것 한글처리 해둘 것
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])