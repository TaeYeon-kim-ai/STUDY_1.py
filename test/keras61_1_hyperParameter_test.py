import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32') /255.


#2. 모델
acriv = 'relu'
node_lstm = 128
node_dnn = 128
drop = 0.3
optimizer = Adam(lr=0.001,epsilon=None)


def build_model(drop = 0.5, optimizer = optimizer) : 
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2]), name = 'input')
    x = LSTM(node_lstm, activation='relu', input_shape = (x_train.shape[1], x_train.shape[2]))(inputs)
    x = Dense(node_dnn, activation= acriv, name = 'hidden1')(x)
    x = BatchNormalization()(x)
    x = Dense(node_dnn, activation= acriv, name = 'hidden2')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)#명시안되있으면 0.51 
    x = Dense(node_dnn, activation= acriv, name = 'hidden3')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation= acriv, name = 'hidden4')(x) #레이어 이름 안겹치게 할 것
    x = BatchNormalization()(x)

    outputs = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics=['acc'], 
                loss = 'categorical_crossentropy')

    return model

def create_hyperparameters() :
    batches = [10, 20, 30, 40, 50]
    oprimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
    ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')
    #러닝레이트, 엑티베이션 넣기 가능
    return {"batch_size" : batches, "optimizer" : oprimizers, "drop" : dropout, }








