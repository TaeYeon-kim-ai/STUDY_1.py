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
print(search.best_score_) #아래 스코어랑 다르니 비교
acc = search.score(x_test, y_test)
print("최종스코어 : ", acc)
print(search.cv_results_)

#케라스 나왔을 땐 케라스가 머신러닝 배꼇다 케라스가 머신러닝 배꼈으나, 케라스의 모델 자체를 사이킷런에서 쓸 수 있게 따진다.
#이 모델은 사이킷런 모델이라고 래핑한다. 케라스모델을 래핑한 모델을 RandomizedSearchCV()안에 넣어준다 인식할 수 있게

# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000277BBF1CD00>
# 0.9577833414077759
# 200/200 [==============================] - 0s 2ms/step - loss: 0.1196 - acc: 0.9639
# 최종스코어 :  0.9639000296592712

# search.cv_results
# 최종스코어 :  0.9681000113487244
# {'mean_fit_time': array([11.23981094,  4.86595337, 13.80927332,  4.93376112,  4.74505615,
#         4.55460127,  5.03626871, 15.90871914,  6.83297483,  7.17614714]), 'std_fit_time': array([0.31823315, 0.08947173, 0.29876036, 0.25327233, 0.05365296,
#        0.10588747, 0.53632744, 0.42123422, 0.25801427, 0.21271081]), 'mean_score_time': array([1.03309584, 0.81448118, 1.39988828, 0.82162484, 0.78697133,
#        0.87605635, 0.93199023, 1.37554733, 0.79212348, 1.10541654]), 'std_score_time': array([0.02701565, 0.07530946, 0.06463341, 0.04995392, 0.0790726 ,
#        0.15299101, 0.19307271, 0.04508051, 0.01059907, 0.03162793]), 'param_optimizer': masked_array(data=['rmsprop', 'adadelta', 'adam', 'adam', 'adadelta',
#                    'adam', 'adadelta', 'adadelta', 'rmsprop', 'adam'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2, 0.1],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[30, 50, 10, 40, 50, 50, 50, 10, 50, 30],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 
# 'adam', 'drop': 0.2, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}], 'split0_test_score': array([0.95859998, 0.13045   , 0.95214999, 0.95644999, 0.1428    ,
#        0.96174997, 0.15885   , 0.2599    , 0.95835   , 0.95050001]), 'split1_test_score': array([0.94375002, 0.2622    , 0.94835001, 0.95534998, 0.23010001,
#        0.95969999, 0.12335   , 0.20565   , 0.9522    , 0.95130002]), 'split2_test_score': array([0.95964998, 0.30864999, 0.95270002, 0.95885003, 0.1498    ,
#        0.95749998, 0.14915   , 0.30544999, 0.95525002, 0.95840001]), 'mean_test_score': array([0.954     , 0.23376666, 0.95106667, 0.95688333, 0.17423334,
#        0.95964998, 0.14378333, 0.257     , 0.95526667, 0.95340002]), 'std_test_score': array([0.00726049, 0.07547693, 0.00193405, 0.00146137, 0.03960693,
#        0.00173541, 0.01498139, 0.04079475, 0.00251076, 0.00355058]), 'rank_test_score': array([ 4,  8,  6,  2,  9,  1, 10,  7,  3,  5])}