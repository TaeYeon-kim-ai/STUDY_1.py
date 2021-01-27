import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x[:5])
# print(y[:10])
# print(x.shape, y.shape) #(442, 10) (442,)
# print(np.max(x), np.min(y))
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #Regressor 회귀 모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #Regressor 회귀 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #Regressor 회귀 모델

#model = LinearRegression()
# result :  -0.5976037028619468
# r2 :  -0.5976037028619468

#model = KNeighborsRegressor() #KNN
# result :  -0.6519673364053473
# r2 :  -0.6519673364053473

#model = DecisionTreeRegressor()
# result :  -0.3232670751992581
# r2 :  -0.3232670751992581

model = RandomForestRegressor()
# result :  -5.604008137602795
# r2 :  -5.604008137602795

#Tensorflow 
# R2 : 0.9430991642272919

#3. 훈련: 머신러닝
from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=20, mode = )
model.fit(x, y)

#4. 평가
y_pred = model.predict(x_test)
#print(y_pred)
# print(y)

# loss, acc = model.evaluate(x_test, y_test) #본래 loss 와 acc이지만
result = model.score(x_test, y_test) #자동으로 evaluate 해서 acc 빼준다.
print('result : ', result)

#accuracy_score
#                  (실데이터, 예측결과 데이터)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)
