# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

def get_column_index(model):
    feature = model.feature_importances_
    feature_list = []
    for i in feature:
        feature_list.append(i)
    feature_list.sort(reverse = True)
 
    result = []
    for j in range(len(feature_list)-len(feature_list)//4):
        result.append(feature.tolist().index(feature_list[j]))
    return result


#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_job = -1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

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
# model1 = RandomForestClassifier(max_depth=4)
# model1 = GradientBoostingClassifier(max_depth=4)
model1 = XGBClassifier(n_job = -1)

#3. 훈련
model1.fit(x1_train, y1_train)

#4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)
print(model1.feature_importances_)
print("acc col정리 : ", acc1)



#DecisionTreeClassifier
# [0.         0.03592329 0.         0.         0.04144514 0.
#  0.1594946  0.         0.         0.         0.0564906  0.33754988
#  0.36909649]
# acc :  0.9444444444444444

#정리후
# [0.40497606 0.11095473 0.30373829 0.07942645 0.04173985 0.05916463
#  0.         0.         0.         0.        ]
# acc col정리 :  0.8888888888888888

#RandomForestClassifier
# 정리전
# [0.14813309 0.02699178 0.01865852 0.02894665 0.01945707 0.0541117
#  0.13440372 0.01550655 0.01882119 0.15159836 0.07967249 0.15611076
#  0.14758812]
# acc :  1.0

#정리후
# [0.10166462 0.1731583  0.13609056 0.21518601 0.18062338 0.07450006
#  0.04379847 0.01494786 0.0263575  0.03367323]
# acc col정리 :  1.0

#GradientBoostingClassifier
#정리전
# [1.16352715e-03 4.90168838e-02 1.75990558e-02 6.88949432e-03
#  6.81616656e-03 2.31756739e-03 1.13743142e-01 1.83440569e-03
#  2.90049341e-05 2.64797114e-01 9.03250117e-03 2.55010430e-01
#  2.71750707e-01]
# acc :  0.9444444444444444

#정리후
# [2.87589639e-01 3.01335430e-01 9.10099295e-03 2.91750935e-01
#  3.69971227e-02 1.13969954e-02 5.51224852e-02 3.17447023e-03
#  3.50659662e-03 2.53333733e-05]
# acc col정리 :  0.9444444444444444

#XGBClassifier
#정리전
# [0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
#  0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
#  0.10358089]
# acc :  1.0

#정리후
# [0.08634423 0.20722187 0.3181203  0.20388341 0.08373912 0.03031781
#  0.0472933  0.01778391 0.00529608 0.        ]
# acc col정리 :  0.9444444444444444