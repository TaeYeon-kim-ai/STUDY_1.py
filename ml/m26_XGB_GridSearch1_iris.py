#데이터 별로 5개 만든다.
import datetime
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor


import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

#파라미터튜닝 적용
parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 66 , shuffle = True)

kfold = KFold(n_splits= 5, shuffle= True)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), parameters, cv = kfold)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("최종 정답률 : ", cross_val_score(y_test, y_pred))

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)

# 최종 정답률 :  1.0

