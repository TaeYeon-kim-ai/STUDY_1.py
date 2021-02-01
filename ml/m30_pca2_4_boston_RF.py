import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#pca
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum : " , cumsum)

#차원 축소 쵀대치 출력
d = np.argmax(cumsum >= 0.99) + 1
print("cumsum >= 0.99 : ", cumsum >=0.99)
print("d : ", d)

#시각화
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

#=================================
pca = PCA(n_components=13, )
x2 = pca.fit_transform(x)
print(x2.shape)

pca_EVR = pca.explained_variance_ratio_

kfold = KFold(n_splits= 5, shuffle=True)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size = 0.8, random_state = 66)

#2. 모델링
model = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold)

#3. 훈련
model.fit(x_train, y_train)

#평가
print("최적의 매개변수 : ", model.best_estimator_)

results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))

#RF
# d :  12
#(506, 13)
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=16)
# 0.7948398213596077
# r2 :  0.7948398213596077

#XGB
# 0.8792669764094689
# r2 :  0.8792669764094689