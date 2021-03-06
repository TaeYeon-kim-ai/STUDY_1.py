# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


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
dataset = load_boston()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
model = DecisionTreeRegressor(max_depth=4)
#model = RandomForestRegressor(max_depth=4)

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
model1 = DecisionTreeRegressor(max_depth=4)
#model1 = RandomForestRegressor(max_depth=4)


#3. 훈련
model1.fit(x1_train, y1_train)

#4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)
print(model1.feature_importances_)
print("acc col정리 : ", acc1)


#DecisionTreeRegressor
# [5.74134125e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  7.65831515e-03 2.96399134e-01 3.70931404e-04 5.95459582e-02
#  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  5.78612249e-01]
# acc :  0.8774175457631728

#정리후
# [0.63868022 0.22083784 0.09401858 0.02595221 0.02051116 0.
#  0.         0.         0.         0.        ]
# acc col정리 :  0.7701633081053628

#RandomForestRegressor
# [3.63942795e-02 9.13236025e-04 2.43977889e-03 3.17871283e-04
#  1.88725861e-02 4.32529090e-01 8.04460541e-03 5.16515275e-02
#  1.42144762e-03 4.40573811e-03 1.22990566e-02 3.96856950e-03
#  4.26742213e-01]
# acc :  0.9043449042642594

#정리후
# [0.38222511 0.47375412 0.06540497 0.02413377 0.02160692 0.00971394
#  0.00427154 0.00856714 0.00649685 0.00382563]
# acc col정리 :  0.8679631482172111