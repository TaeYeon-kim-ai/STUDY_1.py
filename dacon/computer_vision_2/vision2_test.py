import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x = pd.read_csv('C:/data/vision_2/mnist_data/train.csv')
x_test = pd.read_csv('C:/data/vision_2/mnist_data/test.csv')


x_train = x.drop(['id', 'digit'], axis=1)
x_val = x_test.drop(['id'], axis = 1)

dataset = pd.concat([x_train, x_val])
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
print(x) #[22528 rows x 784 columns]
print(y) #Name: letter, Length: 22528, dtype: object

x = x.astype('float32')/255.
print(x.shape) #(22528, 784)
print(y.shape) #(22528,)

x = x.reshape(-1, 28, 28, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle= False ,random_state = 200)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(18022, 28, 28, 1) (4506, 28, 28, 1) (18022,) (4506,)


optimizers = Adam(lr=0.001,epsilon=None)

#2. 모델링
inputs = Input(shape=(28, 28, 1))
x = Conv2D(128, kernel_size = (3,3), activation = 'relu', padding= 'SAME')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(128, 3, activation = 'relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(128, 3, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Conv2D(128, 3, activation = 'relu')(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs = inputs, outputs = outputs)
model.summary()

# model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드

#3.훈련
mc = ModelCheckpoint('C:/data/MC/best_cvision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='loss', patience = 20, mode = 'auto')
lr = ReduceLROnPlateau(monitor='vall_loss', factor= 0.2, patience=10, verbose =1,  mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer= optimizers,  metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size = 32, validation_data = (x_test, y_test), callbacks = [es, lr, mc], verbose= 1)

#평가, 결과
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)





