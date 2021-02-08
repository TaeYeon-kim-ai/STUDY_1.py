#실습

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization
from sklearn.datasets import load_wine
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Adagrad, Adamax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 0)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train) #to_categorical 적용
y_test = to_categorical(y_test) #to_categorical 적용\

#1.1 Minmax
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2.모델
#레이어로 하러면 레이어 하나층을 변수로 지정한 후 쌓기, 노드 수도 변수로 지정하고 수정할 수 있음.(자유)
def build_model(drop = 0.3, optimizer = 'adam', node_dnn = 32, acriv = 'relu', node_ls = 64) :
    inputs = Input(shape = (x_train.shape[1], 1 ), name = 'input')
    x = LSTM(node_ls, activation= 'relu', input_shape = (x_train.shape[1], 1) ,name = 'hidden1')(inputs)
    x = BatchNormalization()(x)
    x = LSTM(node_ls, activation= 'relu', name = 'hidden2')(inputs)
    x = BatchNormalization()(x)
    x = Dense(node_dnn, activation= 'relu', name = 'hidden3')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)#명시안되있으면 0.51
    x = Dense(node_dnn, activation= 'relu', name = 'hidden4')(x)
    x = BatchNormalization()(x)
    x = Dense(node_dnn, activation= 'relu', name = 'hidden5')(x)
    x = BatchNormalization()(x)
    x = Dense(node_dnn, activation= 'relu', name = 'hidden6')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation= 'relu', name = 'hidden7')(x) #레이어 이름 안겹치게 할 것
    outputs = Dense(3, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics=['acc'], 
                loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() : 
    batches = [8, 16, 24]
    optimizer = ['adam', 'rmsprop','adadelta']
    dropout = [0.1, 0.2, 0.3]
    acriv = 'relu'
    node_dnn = [32, 64, 128]
    node_ls = [64, 128, 256]
    #러닝레이트, 엑티베이션 넣기 가능
    return {"batch_size" : batches, "optimizer" : optimizer, "drop" : dropout
            ,"acriv" : acriv, "node_dnn" : node_dnn, "node_ls" : node_ls}

es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')
modelpath = '../data/modelCheckpoint/hyper_14_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')

hyperparameters = create_hyperparameters() #정의된 하이퍼파리미터를 hyperparameters로 저장
model2 = build_model()

#래핑작업 인식할 수 있도록
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor #래핑하기 위한 함수 호출
model2 = KerasClassifier(build_fn=build_model, verbose =1, epochs = 20, 
                        validation_split = 0.2) #build_fn = build_model 우리가 빌드하겠다. 위에 정의된 bulid_model을


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
#search = GridSearchCV(build_model, hyperparameters, cv = 3)

search.fit(x_train, y_train, verbose = 1, callbacks = [es, lr, cp])

print(search.best_params_)#내가 선택한 파라미터 중 가장 좋은거 빼기
print(search.best_estimator_)#전체 중 가장좋은거 선택
print(search.best_score_) #아래 스코어랑 다르니 비교
acc = search.score(x_test, y_test)
print("최종스코어 : ", acc)

#케라스 나왔을 땐 케라스가 머신러닝 배꼇다 케라스가 머신러닝 배꼈으나, 케라스의 모델 자체를 사이킷런에서 쓸 수 있게 따진다.
#이 모델은 사이킷런 모델이라고 래핑한다. 케라스모델을 래핑한 모델을 RandomizedSearchCV()안에 넣어준다 인식할 수 있게

# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000277BBF1CD00>
# 0.9577833414077759
# 200/200 [==============================] - 0s 2ms/step - loss: 0.1196 - acc: 0.9639
# 최종스코어 :  0.9639000296592712

# {'optimizer': 'adam', 'node_dnn': 128, 'drop': 0.1, 'batch_size': 8, 'acriv': 'e'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000002D79F772D60>
# -20.171435356140137
# 13/13 [==============================] - 0s 462us/step - loss: 18.3487 - mse: 18.3487
# 최종스코어 :  -18.348678588867188