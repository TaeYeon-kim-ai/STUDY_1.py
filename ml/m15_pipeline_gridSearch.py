#pipline
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline # 파이프라인/make pipeline  성능은 똑같지만 방식이 다르다
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8)

#1.1 전처리
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# parameters = [
#     {"svc__C" : [1, 10, 100, 1000], "svc__kernel":["linear"]}, #커널에 리니어를 넣을 수 있고 ## 언더바 두개 쓰는건 문법이다.
#     {"svc__C" : [1, 10, 100], "svc__kernel":["rbf"], "svc__gamma":[0.001, 0.0001]}, #커널 rbf
#     {"svc__C" : [1, 10, 100, 10000], "svc__kernel":["sigmoid"], "svc__gamma":[0.001, 0.0001]}
# ]

parameters = [
    {"mal__C" : [1, 10, 100, 1000], "mal__kernel":["linear"]}, #커널에 리니어를 넣을 수 있고 ## 언더바 두개 쓰는건 문법이다.
    {"mal__C" : [1, 10, 100], "mal__kernel":["rbf"], "mal__gamma":[0.001, 0.0001]}, #커널 rbf
    {"mal__C" : [1, 10, 100, 10000], "mal__kernel":["sigmoid"], "mal__gamma":[0.001, 0.0001]}
]                                       #mal은 pipeline 내부 요소 이름지정 이름 + __ + 요소


#2. 모델 pipeline 전처리 합쳐라(파이프로 연결(?)) 전처리1개 + 모델1개
#pipe = make_pipeline(MinMaxScaler(), SVC()) #Grid에서 인식못함
#자동으로 두번 돌지 않음 : for문 써서 결과치 뽑아야함
#전처리를 전체에(위에서) 해버리면 과적합 발생할 수 있음, but train의 scaling 범위가 달라져 성능이 좋아진다. 일반화를 위해서.
#전처리 X : train, test, val, pred 전부해주기 y : 할 필요 없음
#Minmaxscaler 와 SVC모델을 합친다 #MinMax는 따로 import 할 필요 없이 불러와진다.
#crossline이 생길 때 마다 각각 scaling을 해주기 위해 pipeline을 사용한다. 전체에 scaling할 경우 과적합이 발생한다.
#cross_val할 때 마다 train과 val이 구분된다. RS. GR도 CV을 한다. but 그 전에 전처리를 할경우 전체에대해서 scaling적용되는데, 이 경우 val까지 scaling 걸려서 
#val의 데이터(검증하는 부분) 만큼 과적합이 발생한다. val은 Trans까지만 되고 fit은 하면 안된다. pipeline은 자를 때 마다 전처리 해주기에 과적합 문제가 해소될 수 있다.
#pip = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler_A", MinMaxScaler()), ('mal', SVC())]) 

model = GridSearchCV(pipe,parameters, cv= 5)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# SV에서
# val에다가 test를 집어넣는다.