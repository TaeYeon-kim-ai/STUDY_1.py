#실습
#pca를 통해 0.95 이상인거 몇개?
#pca 배운거 다 집어넣고 확인
#안들어가면 리쉐입

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBClassifier

(x_train, _), (x_test, _) = mnist.load_data() #y_train, y_test안쓰겠다.

x = np.append(x_train, x_test, axis=0)

print(x.shape) #(70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape) #(70000, 28, 28)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
#print("cumsum : ", cumsum)

d = np.argmax(cumsum >= 0.95)+1 #95 가능범위 확인
#print("cumsum >= 0.95", cumsum >= 0.95)
print("d : ", d)

#시각화
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

pca = PCA(n_components=154, ) #차원축소
x2 = pca.fit_transform(x)
print(x2.shape)
# d :  154
#(70000, 154)




pca_EVR = pca.explained_variance_ratio_

kfold = KFold(n_splits=4, shuffle=True)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

x_train, x_test = train_test_split(x2, train_size = 0.8, random_state = 0)

#2. 모델링

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold)

#3. 훈련
model.fit(x_train)

#4. 평가
print("최적의 매개변수 : ", model.best_estimator_)
y_pred = model.predict(x_test)
print(y_pred)
