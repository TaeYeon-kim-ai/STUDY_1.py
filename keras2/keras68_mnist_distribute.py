#분산처리
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #1생략 흑백
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,) #1생략 흑백

print(x_train[0])
print(y_train[0])

print("y_trian[0] : ", y_train[0])
print(x_train[0].shape) #(28, 28)

#1.1 데이터 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

#sklearn.onehotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf #GPU분산처리
es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'     
cp = ModelCheckpoint(filepath= modelpath , monitor='val_loss', save_best_only=True, mode = 'auto')

strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()
    )

with strategy.scope() :
    model = Sequential()
    model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
    model.add(Dropout(0.2))
    model.add(Conv2D(200, kernel_size=(3,3), padding = 'SAME'))
    model.add(Dropout(0.2))
    model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
    model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
    model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation= 'relu'))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(10, activation= 'softmax'))

    #3. 컴파일, 훈련

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [es]) #cp

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
print('loss : ', loss)

# y_pred = model.predict(x_test[:10])
# print(y_pred)
# print(y_test[:10])
# print(np.argmax(y_test[:10], axis=-1))

# 시각화
# import  matplotlib.pyplot as plt


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

# plt.show()

# loss :  [0.0957411602139473, 0.975600004196167]

# [7 2 1 0 4 1 4 9 5 9]