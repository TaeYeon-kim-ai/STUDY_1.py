import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링 : 데이터에 따라 결과치 다르게 나타남 다르게 나타남
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
#유사한 특성을 가진 데이터는 유사한 범주에 속하는 경향이 있다는 가정하에 사용. 정규화 해줘야함 , 원을 점차 확대해나가면서.
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
# 데이터가 특정 카테고리에 속할지를 0과 1사이의 연속적인 확률로 예측하는 회귀 알고리즘
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
# 결정나무 : 끝까지 분류, 가지치기. (과적합의 위험)
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN
# 앙상블 기법으로 고안, 훈련과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과 도출에 사용
# 튜닝 쉬움  tree보완, 타깃 예측을 잘함, 대표적인 배깅모델

#머신러닝 모델
#model = LinearSVC()
## result :  0.9666666666666667
## [1 1 1 1]
## [2 2 2 2]
#accuracy_score :  0.9666666666666667

#model = SVC()
## result :  0.9733333333333334
## [0 0 0 0]
## [2 2 2 2]
#accuracy_score :  0.9733333333333334

#model = KNeighborsClassifier()
## result :  0.9666666666666667
## [0 0 0 0]
## [2 2 2 2]
##accuracy_score :  0.9666666666666667

#model = DecisionTreeClassifier()
## result :  1.0
## [0 0 0 0]
## [2 2 2 2]
##accuracy_score :  1.0

#model = RandomForestClassifier()
## result :  1.0
## [0 0 0 0]
## [2 2 2 2]
##accuracy_score :  1.0

model = LogisticRegression()
# result :  0.9733333333333334
# accuracy_score :  0.9733333333333334

# Tensorflow
# acc :  1.0

#3. 훈련:머신러닝 
#다중분류일 경우 : 
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
# model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])
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
