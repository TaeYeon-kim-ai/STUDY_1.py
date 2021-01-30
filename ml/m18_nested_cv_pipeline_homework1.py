# RandomForest 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 diabets

# nested 중첩

#gridSearch 파라미터 100% 다 돌린다. .. 느려;

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
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
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape) #(150, 4)
# print(x[:5])
# print(y.shape) #(150,)

#list형 딕셔너리 제작
#SVC에 들어가있는 파라미터 
parameters = [
    {'randomforestregressor__n_estimators' : [1,2,3], #RF의 n_estimators를 사용
    'randomforestregressor__min_samples_split' : [2,3,4], #RF__
    'randomforestregressor__max_depth' : [1,2,3], #RF__
    'randomforestregressor__max_leaf_nodes' : [2,3,4]} # RF__
]

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)

Kfold = KFold(n_splits = 5, shuffle = True)
for train_index, test_index in Kfold.split(x): 
      print("TRAIN:", train_index, "TEST:", test_index) 
      x_train, x_test = x[train_index], x[test_index] 
      y_train, y_test = y[train_index], y[test_index]
#2 모델
pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor(), parameters, cv = Kfold) # 5회 나눔
score = score = cross_val_score(pipe, x_train ,y_train, cv= Kfold) #5회 나눔

#GS에서 최종으로 5번 분할


#GS에서 나온 5개내에 또 5회씩 분할 
#결과치는 두번째 cross_val에서의 score이 나온다.
#5*5 = 25회

print('교차검증점수 : ', score)



