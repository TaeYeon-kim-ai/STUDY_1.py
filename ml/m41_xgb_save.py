#모델저장

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
model = XGBRegressor(n_estimators = 100, learning_rate = 0.01, n_jobs = -1)

#3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = ['rmse'], 
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

#model_save
import pickle 
'''
# pickle.dump(model, open('../data/xgb_save/m39.pickle.data', 'wb'))#save를 모델로한다. 어디할거냐?
# print('저장완료')

#model_load
model2 = pickle.load(open('../data/xgb_save/m39.pickle.data', 'rb')) #불러오기는 rb  #저장하기는 wb
print('불러오다')
r22 = model2.score(x_test, y_test)
print("r22 : ", r22)

import joblib
joblib.dump(model,"../data/xbg_save/m39.joblib.data")
print('저장하다')
# model2 = joblib.load('../data/xgb_save/m39.joblib.data')
# print('불러오다')
# r22 = model.score(x_test, y_test)
# print("r22 : ", r22)
'''

# #XGB자체 save
# model.save_model('../data/xgb_save/m39.xgb.data')
# print('저장하다')

#XGB load
model2 = XGBRegressor()
model2.load_model('../data/xgb_save/m39.xgb.data')
r22 = model2.score(x_test, y_test)
print("r22 :", r22)





