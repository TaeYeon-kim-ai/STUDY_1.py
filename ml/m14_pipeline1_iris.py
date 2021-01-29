#pipline
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
                                                                            #CV : cross_validation까지 하겠다.
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline # 파이프라인/make pipeline  성능은 똑같지만 방식이 다르다
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
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8)

#1.1 전처리
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 pipeline 전처리 합쳐라(파이프로 연결(?)) 전처리1개 + 모델1개
#model = Pipeline([("scaler_A", MinMaxScaler()), ('SVC_A', SVC())]) #Minmaxscaler 와 SVC모델을 합친다 #MinMax는 따로 import 할 필요 없이 불러와진다.
model = make_pipeline(MinMaxScaler(), SVC())
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

