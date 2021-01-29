# 실습
#RandomSearchdhk와 GS와 Pipeline을 엮어라!
#모델은 RandomFotest
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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 66, shuffle = True)


parameters = [
    {"mal__C" : [1, 10, 100, 1000], "mal__kernel":["linear"]}, #커널에 리니어를 넣을 수 있고 ## 언더바 두개 쓰는건 문법이다.
    {"mal__C" : [1, 10, 100], "mal__kernel":["rbf"], "mal__gamma":[0.001, 0.0001]}, #커널 rbf
    {"mal__C" : [1, 10, 100, 10000], "mal__kernel":["sigmoid"], "mal__gamma":[0.001, 0.0001]}
]                                       #mal은 pipeline 내부 요소 이름지정 이름 + __ + 요소

kfold = KFold(n_splits= 5, shuffle= True)

pipe = Pipeline([("scaler_A", MinMaxScaler()), ('mal', SVC())]) 

cv = ([RandomizedSearchCV(), GridSearchCV()])

for i in cv :
    model = cv(pipe, parameters, cv = kfold)

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))
