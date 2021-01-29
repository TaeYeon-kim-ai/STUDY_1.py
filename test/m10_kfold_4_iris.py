import numpy as np
import pandas as pd
import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
#SVM은 각 데이터가 떨어진 거리를 가장 극대화 시켜 찾는 방법.
from sklearn.svm import LinearSVC, SVC # SVM(Support Vector Machine)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header = 0, index_col = 0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
x = x.to_numpy()
y = y.to_numpy()

#1.1 parameters 딕셔너리
#파라미터 튜닝
parameters = [
    {'n_estimators' : [1,2,3],
    'max_depth' : [1,2,3],
    'min_samples_leaf' : [1,2,3],
    'min_samples_split' : [1,2,3],
    'n_jobs' : [-1]
    }
]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
    random_state = 66, shuffle = True)

#2모델링
kfold = KFold(n_splits=4, shuffle=True)

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv= kfold)

#3.훈련
print("start time - {}".format(datetime.datetime.now().strftime("%Y%m% %H%M%S")))
model.fit(x_train, y_train)
print("End time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


#4.평가, 예측
print("최적의 매개변수", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률", accuracy_score(y_test, y_pred))

scores = cross_val_score(model, x, y, cv = kfold)
print('scores : ', scores)







