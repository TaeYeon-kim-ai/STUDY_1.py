
#pipline
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8)

kfold = KFold(n_splits=5, shuffle=True)
#1.1 딕셔너리 작성

parameter = [
    {'n_estimators' : [10, 100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20],
    'n_jobs' : [-1]}
]

#2. 모델 pipeline 전처리 합쳐라(파이프로 연결(?)) 전처리1개 + 모델1개
model = Pipeline([("scaler_A", MiinMaxScaler()), ('SVC_A', SVC())]) #Minmaxscaler 와 SVC모델을 합친다 #MinMax는 따로 import 할 필요 없이 불러와진다.
model = make_pipeline(StandardScalecr(), RandomizedSearchCV(RandomForestClassifier(), parameter, cv = kfold))

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = kfold)
print('scores : ', scores)

#MinMax
# 0.9298245614035088
# 최종 정답률 :  0.9298245614035088
# scores :  [0.98245614 0.96491228 0.93859649 0.92982456 0.96460177]

#Standar
# 0.9385964912280702
# 최종 정답률 :  0.9385964912280702
# scores :  [0.96491228 0.96491228 0.96491228 0.93859649 0.9380531 ]