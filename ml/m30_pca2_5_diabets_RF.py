import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor, plot_importance
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor


datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #자세히 확인 explained_variance_rati
print("cumsum : ", cumsum) #max = 1
# cumcum :  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
# 0.94794364 0.99131196 0.99914395 1.        ] #cumsum작은것부터 하나씩 더해짐
# 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수

d = np.argmax(cumsum >= 0.99)+1 # 0.95중 최댓값 중 +1되는 걸로 써라(실질적인 가장 큰 값)
print("cumsum >=0.99", cumsum >=0.99)
print("d : ", d)
# cumsum >=0.95 [False False False False False False False  True  True  True]
# d :  8

#시각화
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

#헷갈림 주의
#feature importance : feature자체의 중요도를 찾는 것
#pca : 압축을 했을 때 몇개쓰면 좋은거냐 데이터 그대로쓰면 100%지만 줄일수록 손실이 있음. but 손실률 을 말하는것이지 score자체를 말하는건 아님 95%이상 기준?(개인)
#      - 원 데이터에서 데이터가 변형됨 / 전처리와 비슷한 개념.y 값과 매칭되는건 바뀌지 않음.

#데이터 - 전처리 - pca - 모델링 - 훈련 - 평가

pca = PCA(n_components = 9, ) #n_components 압축 열 수 지정
x2 = pca.fit_transform(x) #pca를 핏트랜스폼 (x)
print(x2.shape) #(442, 7) #컬럼 재구성

# #압축된 것 중에서 어떤 피쳐가 중요한지
pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR) #컬럼 10개를 7개로 압축시켜서
# print(sum(pca_EVR))

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size = 0.8, random_state = 0)

#2. 모델링
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가

print("최적의 매개변수 : ", model.best_estimator_)

results = model.score(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))

#RF
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=8, min_samples_split=16,
#                       n_estimators=10)
# d :  8
# (442, 9)
# 0.27057671434697794
# r2 :  0.27057671434697794

#XGB
# 0.2308722822086997
# r2 :  0.2308722822086997