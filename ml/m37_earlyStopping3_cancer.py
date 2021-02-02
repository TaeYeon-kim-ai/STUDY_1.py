import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

#2. 모델링
model = XGBClassifier(n_estimators = 100, learning_rate = 0.01, n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric=['error', 'logloss'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=5)

#4. 평가
result1 = model.score(x_test, y_test)
print("result1 : ", result1)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

result2 = model.evals_result()
print("result2 : ", result2)

# result1 :  0.9722222222222222
# acc :  0.9722222222222222