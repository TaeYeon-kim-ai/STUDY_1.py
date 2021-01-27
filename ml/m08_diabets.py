import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x[:5])
# print(y[:10])
# print(x.shape, y.shape) #(442, 10) (442,)
# print(np.max(x), np.min(y))
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 0)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor #Regressor 회귀 모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #Regressor 회귀 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #Regressor 회귀 모델

# model = LinearRegression()
# result :  -193.84297898963447
# r2 :  -193.84297898963447

model = KNeighborsRegressor() #KNN
# result :  0.19300609798881352
# r2 :  0.19300609798881352

# model = DecisionTreeRegressor()
# result :  -0.7761782309505243
# r2 :  -0.7761782309505243

# model = RandomForestRegressor()
# result :  0.029661470514159904
# r2 :  0.029661470514159904

#Tensorflow
# r2 : 0.5128401315682825

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
