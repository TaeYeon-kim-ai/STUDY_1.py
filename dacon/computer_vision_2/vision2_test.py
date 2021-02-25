import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.eager.monitoring import Sampler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

#1. 데이터

#ImageDatagenrtator & data augmentation
x_train = np.load('C:/data/vision_2/dirty_mnist_npy/vision2_train_x.npy')/255. #x 1
y = pd.read_csv('C:/data/vision_2/dirty_mnist_2nd_answer.csv')
y = y.drop(['index'], axis=1).values  # 숨겨진 숫자 값
y_train = y.iloc[:,:]
test = np.load('C:/data/vision_2/dirty_mnist_npy/vision2_test_x.npy')/255. #x 1
submission = pd.read_csv('C:/data/vision_2/sample_submission.csv')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle= True, random_state=256)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
'''

inputs = Input(shape=(256, 256, 3)) 
x = Conv2D(128, 4, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Conv2D(128, 2, activation='relu')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'binary_crossentropy', optimizer=optimizers, metrics=['acc'])    
model.summary()

#model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드

#3훈련
optimizers = Adam(lr=0.001,epsilon=None)
mc = ModelCheckpoint('C:/data/MC/best_cvision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='loss', patience = 20, mode='auto')
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='auto')
model.fit(x_train, y_train, epochs = 2, batch_size = 64 ,validation_data = (x_val, y_val), callbacks=[es, lr, mc])

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
y_pred = model.predict(x_pred)
print(y_pred)
y_pred_sub = np.where(y_pred < 0.5, 0, 1)
print(y_pred_sub)

submission[i] = y_pred_sub
    
submission.to_csv('C:/data/vision_2/submission_2021.02.25_1.csv', index=False)



'''