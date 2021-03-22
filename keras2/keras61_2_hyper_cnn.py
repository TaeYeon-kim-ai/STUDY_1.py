#CNN으,로 수정
#파라미터 변경할 것
#필수 : 노드의 갯수 (반드시 넣어야 할 것 : 노드의 수 , 나머지는 변수화)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터/ 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.


#2.모델
acriv = 'relu'
node_cnn = 32
node_dnn = 128
kernel = (2,2)
drop = 0.3
optimizer = 'adam'

#레이어로 하러면 레이어 하나층을 변수로 지정한 후 쌓기, 노드 수도 변수로 지정하고 수정할 수 있음.(자유)
def build_model(drop = 0.5, optimizer = optimizer) :
    inputs = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]), name = 'input')
    x = Conv2D(node_cnn, kernel_size = (3,3), strides=1, padding = 'SAME' , activation='relu', input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))(inputs)
    x = Conv2D(node_cnn, kernel, activation= acriv, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Conv2D(node_cnn, kernel, activation= acriv, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Flatten()(x)

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
    return {"batch_size" : batches, "optimizer" : oprimizers, "drop" : dropout, "es" : EarlyStopping, "lr" : ReduceLROnPlateau}

hyperparameters = create_hyperparameters() #정의된 하이퍼파리미터를 hyperparameters로 저장
model2 = build_model()

#래핑작업 인식할 수 있도록
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor #래핑하기 위한 함수 호출
model2 = KerasClassifier(build_fn=build_model, verbose =1, epochs = 3, 
                        validation_split = 0.2, callback = [es, lr]) #build_fn = build_model 우리가 빌드하겠다. 위에 정의된 bulid_model을


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
#search = GridSearchCV(build_model, hyperparameters, cv = 3)


search.fit(x_train, y_train, verbose = 1)

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


