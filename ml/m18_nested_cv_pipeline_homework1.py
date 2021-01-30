# RandomForest 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 diabets

# nested 중첩

#gridSearch 파라미터 100% 다 돌린다. .. 느려;

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
# from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
# from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

#kfold
Kfold = KFold(n_splits = 5, shuffle = True)

parameters = [
    {'randomforestregressor__n_estimators' : [1,10, 100], #RF의 n_estimators를 사용
    'randomforestregressor__min_samples_split' : [20,30,4], #RF__
    'randomforestregressor__max_depth' : [11,12,13], #RF__
    'randomforestregressor__max_leaf_nodes' : [12,13,14]} # RF__
]  

#2 모델

pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
        
model = RandomizedSearchCV(pipe, parameters, cv = Kfold)

score = cross_val_score(model, x ,y, cv= Kfold)

print('교차 검증점수 : ', score)

# 교차 검증점수 :  [0.41784982 0.43283835 0.386734   0.45417541 0.53655542]

# for train_index, test_index in Kfold.split(x): 
#     print("TRAIN:", train_index, "TEST:", test_index) 
#     x_train, x_test = x[train_index], x[test_index] 
#     y_train, y_test = y[train_index], y[test_index]
