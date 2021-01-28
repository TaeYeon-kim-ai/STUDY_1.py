#RandomizedSearchCV = 랜덤해서 선택, 빠르다. 
#(과제) RandomizedSearchCV는 어떻게 Random하길래 빠른가?
#gridSearch 파라미터 100% 다 돌린다. .. 느려;

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#1. 데이터
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)#헤더랑 인덱스 잡아주기
x = dataset.iloc[:,:-1] #150, 4
y = dataset.iloc[:,-1] # 150,
print(x.shape, y.shape)

#list형 딕셔너리 제작
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel":["linear"]}, #커널에 리니어를 넣을 수 있고
    {"C" : [1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #커널 rbf
    {"C" : [1, 10, 100, 10000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)

Kfold = KFold(n_splits = 5, shuffle = True) # 훈련, 검증 5개 만들어준다

model = RandomizedSearchCV(SVC(), parameters, cv = Kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) #90번 한 것 중에 가장 좋은거 빼줌
print('최종정답률', accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = Kfold) 
print('scores : ', scores)

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최종정답률 0.9666666666666667
# scores :  [1.         1.         0.96666667 0.93333333 1.        ]

# Tensorflow
# acc :  1
