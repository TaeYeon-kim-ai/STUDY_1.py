from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import  numpy as np
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66 )

#2. 모델
model = XGBRegressor(n_estimators = 700, learning_rate = 0.01, n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = ['logloss', 'rmse'], 
        eval_set= [(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds= 5
)
#4. 평가
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred) #r2잡을 때 원 데이터가 앞으로 가게?
print("r2 : ", r2)

# aaa :  0.9329663244922279
# r2 :  0.9329663244922279

print("=======================")
results = model.evals_result() # 터미널에서 훈련 셋 지표(rmse)가 줄어드는 과정 표기
print(results)


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
ax.plot(x_axis, results['validation_0']['rmse'], label = 'train')
ax.plot(x_axis, results['validation_1']['rmse'], label = 'test')
ax.legend()
plt.ylabel('rmse')
plt.title('XGBoost RMSE')
plt.show()










