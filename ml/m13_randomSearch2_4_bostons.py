import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import warnings
import datetime

# 경고처리
warnings.filterwarnings('ignore')

#1. 데이터
x = np.load('../data/npy/boston_x.npy')
y = np.load('../data/npy/boston_y.npy')
print(x.shape, y.shape)# (506, 13) (506,)

#1.1 parameters 딕셔너리 제작
parameter = [
    {'n_estimators' : [10, 100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]},
    
    {'n_estimators' : [100, 200, 300, 500], 
    'max_depth' :  [6, 7, 8, 10], 
    'min_samples_leaf' : [3, 5, 11, 12,13, 14, 15], 
    'min_samples_split' : [2, 4, 6, 8, 10], 
    'n_jobs' : [-1]},
    
    {'n_estimators' : [100, 200, 300, 500], 
    'max_depth' :  [6, 9, 12, 15, 18, 21, 24], 
    'min_samples_leaf' : [3, 6, 7, 10, 12], 
    'min_samples_split' : [2, 4, 8, 16], 
    'n_jobs' : [-1]}
]

#RandomForestClassifier
#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 0, shuffle = True)

#2. 모델링
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestRegressor(), parameter, cv = kfold)

#3.훈련
print ("Start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
model.fit(x_train, y_train)
print ("End time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#4. 평가
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률 : ", r2_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = kfold)
print('scores : ', scores)
# Start time - 2021-01-28 17:12:15
# End time - 2021-01-28 17:12:24
# 최적의 매개변수 :  RandomForestRegressor(max_depth=24, min_samples_leaf=3, min_samples_split=8,
#                       n_estimators=300, n_jobs=-1)
# 최종 정답률 :  0.7030140509543932