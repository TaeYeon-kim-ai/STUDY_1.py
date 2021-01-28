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
#첫번 째 돌아갈 때 C라는 파라미터에 1주고 리니어로 한바퀴돌고 -- 결과
#두번 째 10주고 linear에서 1바퀴
#셋 째 100주고 linear에서 1바퀴  __ 3회  
#_총 18회 돈다. 파라미터 하나하나에 맞춰서 parameters내 값을 돌리겠다.
#케라스에 있는 파라미터 함수에 넣은다음 돌릴 것

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)

Kfold = KFold(n_splits = 5, shuffle = True) # 훈련, 검증 5개 만들어준다
#2. 모델링 : 데이터에 따라 결과치 다르게 나타남 다르게 나타남
#SVC모델을 GridSearchCV로 감사겠다. 이 자체로 모델로 사용

model = GridSearchCV(SVC(), parameters, cv = Kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) #90번 한 것 중에 가장 좋은거 빼줌
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
#3. 훈련:머신러닝 
#다중분류일 경우 : 
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
# model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [es])
model.fit(x, y)

#4. 평가
# loss, acc = model.evaluate(x_test, y_test) #본래 loss 와 acc이지만
result = model.score(x_test,y_test) #자동으로 evaluate 해서 acc 빼준다.
print('result : ', result)

y_pred = model.predict(x_test)
# print(y_pred)
# print(y)

#accuracy_score
#                  (실데이터, 예측결과 데이터)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)
'''