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
model = XGBClassifier(n_estimators = 500, learning_rate = 0.01, n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric=['error','logloss'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=5)

#4. 평가
result1 = model.score(x_test, y_test)
print("result1 : ", result1)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

results = model.evals_result()
print("result2 : ", results)

# result1 :  0.9722222222222222
# acc :  0.9722222222222222

#5. 시각화
import  matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label = 'train')
ax.plot(x_axis, results['validation_1']['logloss'], label = 'test')
ax.legend()
plt.ylabel('log Loss')
plt.title('XGBoost Log Loss')
#plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label = 'train')
ax.plot(x_axis, results['validation_1']['error'], label = 'test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost ERROR')
plt.show()
