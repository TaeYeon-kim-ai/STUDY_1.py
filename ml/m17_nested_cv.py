# nested 중첩

#gridSearch 파라미터 100% 다 돌린다. .. 느려;

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
# from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
# from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
# from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

#list형 딕셔너리 제작
#SVC에 들어가있는 파라미터 
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel":["linear"]}, #커널에 리니어를 넣을 수 있고
    {"C" : [1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #커널 rbf
    {"C" : [1, 10, 100, 10000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)

Kfold = KFold(n_splits = 5, shuffle = True)
#2 모델
model = GridSearchCV(SVC(), parameters, cv = Kfold)
#GS에서 최종으로 5번 분할
score = cross_val_score(model, x_train ,y_train, cv= Kfold)
#GS에서 나온 5개내에 또 5회씩 분할 
#결과치는 두번째 cross_val에서의 score이 나온다.
#5*5 = 25회

print('교차검증점수 : ', score)

'''
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) 
print('최종정답률', accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = Kfold) 
print('scores : ', scores)

aaa = model.score(x_test, y_test)
print(aaa)

# 최적의 매개변수 :  SVC(C=1000, kernel='linear')
# 최종정답률 1.0
# scores :  [1.         0.93333333 0.93333333 1.         1.        ]
# 1.0

# Tensorflow
# acc :  1
'''