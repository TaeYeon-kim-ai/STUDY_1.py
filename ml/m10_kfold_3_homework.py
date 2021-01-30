# 실습 또는 과제!!
#kfold validation 적용
#train_test나눈 다음에 발리데이션 하지말고
# kfold한 후에 train_test_split 사용
# 기존 : kfold -----> train_test  5등분 - 2등분
# 결과 : train_test -----> kfold  2등분 - 5등분
# 결과 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score#교차 검증값
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

#kfold
Kfold = KFold(n_splits = 5, shuffle = True) # 훈련, 검증 5개 만들어준다 # train_test_split시 validaton
for train_index, test_index in Kfold.split(x): 
      print("TRAIN:", train_index, "TEST:", test_index) 
      x_train, x_test = x[train_index], x[test_index] 
      y_train, y_test = y[train_index], y[test_index]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state = 77, shuffle = True, train_size = 0.8)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
     

#2. 모델링 : 데이터에 따라 결과치 다르게 나타남 다르게 나타남
model = LinearSVC()

scores = cross_val_score(model, x_train, y_train, cv = Kfold) 
print('scores : ', scores)
# scores :  [0.96666667 0.96666667 0.93333333 0.93333333 0.93333333] #각 등분마다의 결과값
# scores :  [0.95833333 1.         1.         0.91666667 1.        ] # train_test_split 적용

# Tensorflow
# acc :  1.0

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