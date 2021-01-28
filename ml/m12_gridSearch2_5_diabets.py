import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score,r2_score
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
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

#parameters 딕셔너리 제작
parameters = [
    {'n_estimators' : [10, 100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]},
    
    {'n_estimators' : [100, 200, ], 
    'max_depth' :  [6, 7, 8], 
    'min_samples_leaf' : [3, 5], 
    'min_samples_split' : [2, 4, 6, 8, 10], 
    'n_jobs' : [-1]},
    
    {'n_estimators' : [100, 200, 300, 500], 
    'max_depth' :  [18, 21, 24], 
    'min_samples_leaf' : [7, 10, 12], 
    'min_samples_split' : [8, 16], 
    'n_jobs' : [-1]}
    ]

 #총 15회 훈련 늘리기
# RandomForestRegressor 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestRegressor(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = parameters, cv = 3, n_jobs = -1)

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
Kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(RandomForestRegressor(), parameters, cv = Kfold) 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) #90번 한 것 중에 가장 좋은거 빼줌
print('최종정답률', r2_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = Kfold) 
print('scores : ', scores)

aaa = model.score(x_test, y_test)
print('score : ', aaa)

# 최적의 매개변수 :  RandomForestRegressor(max_depth=7, min_samples_leaf=5, min_samples_split=8,
#                       n_jobs=-1)
# 최종정답률 0.4697477461443865

#Tensorflow
# r2 : 0.5128401315682825