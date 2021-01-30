#pipline
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline # 파이프라인/make pipeline  성능은 똑같지만 방식이 다르다
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8)

parameters = [
    {'randomforestregressor__n_estimators' : [1,2,3], #RF의 n_estimators를 사용
    'randomforestregressor__min_samples_split' : [2,3,4], #RF__
    'randomforestregressor__max_depth' : [1,2,3], #RF__
    'randomforestregressor__max_leaf_nodes' : [2,3,4]} # RF__
]


scales = [MinMaxScaler(), StandardScaler()]
search = [RandomizedSearchCV, GridSearchCV]


for i in scales : #i안에 scales로 정의된 함수들을 넣는다 전처리
    pipe = make_pipeline(i, RandomForestRegressor()) #RFC로 모델구성
    for j in search :
        model = j(pipe,parameters, cv= 5) #class_val)score출력, cv = 5로 구성 데이터를 5번쪼개서 훈련
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        #print("최적의 매개변수 : ", model.best_estimator_)
        print(f'score_{i}_{j.__name__} : ', model.score(x_test,y_test))


# score_MinMaxScaler()_RandomizedSearchCV :  0.7297574994902172
# score_MinMaxScaler()_GridSearchCV :  0.7397969309537162
# score_StandardScaler()_RandomizedSearchCV :  0.7294508166474122
# score_StandardScaler()_GridSearchCV :  0.7206565329746173