import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)
print(np.max(x), np.min(y))
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
#유사한 특성을 가진 데이터는 유사한 범주에 속하는 경향이 있다는 가정하에 사용. 정규화 해줘야함 , 원을 점차 확대해나가면서.
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
# 데이터가 특정 카테고리에 속할지를 0과 1사이의 연속적인 확률로 예측하는 회귀 알고리즘
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

#머신러닝 모델
# model = LinearSVC()
## result :  0.4298245614035088
## accuracy_score :  0.4298245614035088

# model = SVC()
# # result :  0.6403508771929824
# # accuracy_score :  0.6403508771929824

# model = KNeighborsClassifier()
# # result :  0.6403508771929824
# # accuracy_score :  0.6403508771929824

# model = DecisionTreeClassifier()
# # result :  0.8245614035087719
# # accuracy_score :  0.8245614035087719

# model = RandomForestClassifier()
# # result :  0.7543859649122807
# # accuracy_score :  0.7543859649122807

model = LogisticRegression()
# result :  0.5614035087719298
# accuracy_score :  0.5614035087719298

#Tensorflow 
#acc : 0.9912280440330505

#3. 훈련: 머신러닝
from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=20, mode = )
model.fit(x, y)

#4. 평가
# loss, acc = model.evaluate(x_test, y_test) #본래 loss 와 acc이지만
result = model.score(x_test, y_test) #자동으로 evaluate 해서 acc 빼준다.
print('result : ', result)

y_pred = model.predict(x_test)
#print(y_pred)
# print(y)

#accuracy_score
#                  (실데이터, 예측결과 데이터)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)
