# 실습
#RandomSearchdhk와 GS와 Pipeline을 엮어라!
#모델은 RandomFotest
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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 66, shuffle = True)

kfold = KFold(n_splits= 5, shuffle= True)
parameters = [
    {'n_estimators' : [10, 100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]},
    
    {'n_estimators' : [100, 300, 500], 
    'max_depth' :  [6, 7, 10], 
    'min_samples_leaf' : [3, 14, 15], 
    'min_samples_split' : [2, 4, 6, 8, 10], 
    'n_jobs' : [-1]},
    
    {'n_estimators' : [100, 500], 
    'max_depth' :  [6, 9, 24], 
    'min_samples_leaf' : [10, 12], 
    'min_samples_split' : [2, 4, 8, 16], 
    'n_jobs' : [-1]}
]
pipe = Pipeline(make_pipeline(MinMaxScaler(), RandomForestClassifier()))
#model = RandomizedSearchCV(pipe, parameters, cv = kfold)
model = GridSearchCV(pipe, parameters, cv = kfold)

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))
