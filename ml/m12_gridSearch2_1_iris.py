import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier #K-최근접 이웃 #classifier 분류모델 model = KNeighborsClassifier KNN
from sklearn.linear_model import LogisticRegression #Logistic = 분류모델
from sklearn.tree import DecisionTreeClassifier #classifier 분류모델 model = DecisionTreeClassifier DTN
from sklearn.ensemble import RandomForestClassifier #classifier 분류모델 model = RandomForestClassifier RFN

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

#parameters 딕셔너리 제작
parameters = [
    {'n_estimators' : [10, 100],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]},
    
    {'n_estimators' : [100, 200, 300, 500], 
    'max_depth' :  [6, 7, 8, 10], 
    'min_samples_leaf' : [3, 5, 11, 12,13, 14, 15], 
    'min_samples_split' : [2, 4, 6, 8, 10], 
    'n_jobs' : [-1]},
    
    {'n_estimators' : [100, 200, 300, 500], 
    'max_depth' :  [6, 9, 12, 15, 18, 21, 24], 
    'min_samples_leaf' : [3, 6, 7, 10, 12], 
    'min_samples_split' : [2, 4, 8, 16], 
    'n_jobs' : [-1]}
    ]
# n_estimators	
# - 결정트리의 갯수를 지정
# - Default = 10
# - 무작정 트리 갯수를 늘리면 성능 좋아지는 것 대비 시간이 걸릴 수 있음

# min_samples_split	
# - 노드를 분할하기 위한 최소한의 샘플 데이터수
# → 과적합을 제어하는데 사용
# - Default = 2 → 작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가

# min_samples_leaf	
# - 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수
# - min_samples_split과 함께 과적합 제어 용도
# - 불균형 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 작게 설정 필요

# max_features	
# - 최적의 분할을 위해 고려할 최대 feature 개수
# - Default = 'auto' (결정트리에서는 default가 none이었음)
# - int형으로 지정 →피처 갯수 / float형으로 지정 →비중
# - sqrt 또는 auto : 전체 피처 중 √(피처개수) 만큼 선정
# - log : 전체 피처 중 log2(전체 피처 개수) 만큼 선정

# max_depth	
# - 트리의 최대 깊이
# - default = None
# → 완벽하게 클래스 값이 결정될 때 까지 분할 또는 데이터 개수가 min_samples_split보다 작아질 때까지 분할
# - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요

# max_leaf_nodes	리프노드의 최대 개수
    

    

 #총 15회 훈련 늘리기
# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = parameters, cv = 3, n_jobs = -1)

#kfold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
Kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv = Kfold) 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) #90번 한 것 중에 가장 좋은거 빼줌
print('최종정답률', accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = Kfold) 
print('scores : ', scores)

aaa = model.score(x_test, y_test)
print('score : ', aaa)

# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=7)
# 최종정답률 0.9
# scores :  [0.86666667 0.93333333 0.93333333 1.         1.        ]
# score :  0.9

# Tensorflow
# acc :  1
