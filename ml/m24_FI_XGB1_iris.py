# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import datetime

#xgb 사용
from xgboost import XGBClassifier


def get_column_index(model):
    feature = model.feature_importances_ # feature_importances
    feature_list = [] #list
    for i in feature: # i 로 반환
        feature_list.append(i)
    feature_list.sort(reverse = True) #역으로 배열
 
    result = [] 
    for j in range(len(feature_list)-len(feature_list)//4): # 1/4 추출 전체에서 제거
        result.append(feature.tolist().index(feature_list[j]))
    return result


#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
#model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_job = -1) 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)

#n_job = -1 time 0:00:00.029920
#n_job = 8 time 0:00:00.035905
#n_job = 1 time 0:00:00.030917
#n_job = 2 time 0:00:00.038897
#n_job = 3 time 0:00:00.029921
#n_job = 4 time 0:00:00.037899
#n_job = 5 time 0:00:00.036902
#n_job = 6 time 0:00:00.034907
#n_job = 7 time 0:00:00.030917

#5. 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model): 
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Improtances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

#=================================================================
#함수 적용 컬럼 추출
result = get_column_index(model)
print(result)
x1 = pd.DataFrame(dataset.data, columns = dataset.feature_names)
x1 = x1.iloc[:, [x for x in result]]
y1 = dataset.target

x1 = x1.values

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.2, random_state = 77
)

#2. 모델
# model1 = DecisionTreeClassifier(max_depth=4)
#model1 = RandomForestClassifier(max_depth=4)
#model1 = GradientBoostingClassifier(max_depth=4)
model1 = XGBClassifier(n_job = -1, use_label_encoder= False) #ABCD 를 1,2,3,4로 변환해준다. 라벨인코더 하지 않을거면 False


#3. 훈련
model1.fit(x1_train, y1_train, eval_metric='logloss') #매트릭스 안쓸려면 logloss로 써라

#4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)
print(model1.feature_importances_)
print("acc col정리 : ", acc1)


#n_job = -1




#DecisionTreeClassifier
#정리전
# [0.0125026  0.         0.03213177 0.95536562]
# acc :  0.9333333333333333


#정리후
# [0.44369011 0.53961888 0.01669101]
# acc col정리 :  0.9

#RandomForestClassifier
#정리전
# [0.08142011 0.02056809 0.41412333 0.48388846]
# acc :  0.9666666666666667

#정리후
# [0.3916402  0.37551523 0.23284457]
# acc col정리 :  0.8666666666666667

#GradientBoostingClassifier
#정리전
# [0.00767652 0.00935898 0.25834707 0.72461743]
# acc :  0.9333333333333333

#정리후
# [0.84998508 0.13328116 0.01673376]
# acc col정리 :  0.9

#XGBClassifier
#정리전
#[0.01835513 0.0256969  0.62045246 0.3354955 ]
# acc :  0.9

#정리후
# [0.44489893 0.5354024  0.01969868]
# acc col정리 :  0.8333333333333334




