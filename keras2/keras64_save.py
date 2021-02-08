#가중치 저장할 것
#1. model.save()
#2. pickle 쓸것

#61카피해서
#model.cv_results를 붙여서 완성
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터/ 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


#2.모델
#레이어로 하러면 레이어 하나층을 변수로 지정한 후 쌓기, 노드 수도 변수로 지정하고 수정할 수 있음.(자유)
def build_model(drop = 0.5, optimizer = 'adam') :
    inputs = Input(shape = (28*28), name = 'input')
    x = Dense(512, activation= 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation= 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)#명시안되있으면 0.5
    x = Dense(128, activation= 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics=['acc'], 
                loss = 'categorical_crossentropy')

    return model

def create_hyperparameters() : 
    batches = [10, 20, 30, 40, 50]
    oprimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : oprimizers, "drop" : dropout}

hyperparameters = create_hyperparameters() #정의된 하이퍼파리미터를 hyperparameters로 저장
model2 = build_model()

#래핑작업 인식할 수 있도록
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor #래핑하기 위한 함수 호출
model2 = KerasClassifier(build_fn=build_model, verbose =1) #build_fn = build_model 우리가 빌드하겠다. 위에 정의된 bulid_model을


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
#search = GridSearchCV(build_model, hyperparameters, cv = 3)


search.fit(x_train, y_train, verbose = 1)
print(search.best_params_)#내가 선택한 파라미터 중 가장 좋은거 빼기
print(search.best_estimator_)#전체 중 가장좋은거 선택

#====================================================
# 모델저장
import pickle
pickle.dump(model2, open('C:/data/keras_save/keras64_save.pickle.data', 'rb'))#save
print('저장완료')

# model3 = pickle.load(open('C:/data/keras_save/keras64_save.pickle.data', 'rb'))#open
# print('불러오다')
#====================================================

print(search.best_score_) #아래 스코어랑 다르니 비교
acc = search.score(x_test, y_test)
print("최종스코어 : ", acc)
print(model2.cv_results)



#케라스 나왔을 땐 케라스가 머신러닝 배꼇다 케라스가 머신러닝 배꼈으나, 케라스의 모델 자체를 사이킷런에서 쓸 수 있게 따진다.
#이 모델은 사이킷런 모델이라고 래핑한다. 케라스모델을 래핑한 모델을 RandomizedSearchCV()안에 넣어준다 인식할 수 있게

# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000277BBF1CD00>
# 0.9577833414077759
# 200/200 [==============================] - 0s 2ms/step - loss: 0.1196 - acc: 0.9639
# 최종스코어 :  0.9639000296592712