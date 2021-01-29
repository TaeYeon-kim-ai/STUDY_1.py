# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
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
dataset = load_diabetes()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
#model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor(max_depth=4)
model = GradientBoostingRegressor(max_depth=4)

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
#model1 = DecisionTreeRegressor(max_depth=4)
#model1 = RandomForestRegressor(max_depth=4)
model1 = GradientBoostingRegressor(max_depth=4)


#3. 훈련
model1.fit(x1_train, y1_train)

#4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)
print(model1.feature_importances_)
print("acc col정리 : ", acc1)


#DecisionTreeRegressor
# [0.03400704 0.         0.26623557 0.11279298 0.         0.
#  0.01272153 0.00124721 0.51986371 0.05313196]
# acc :  0.33339660919782466

#정리후
# [0.51214594 0.33393422 0.0856219  0.01202884 0.01669709 0.03530176
#  0.         0.00427026]
# acc col정리 :  0.38349331181632773

#RandomForestRegressor
# [0.03249957 0.00236945 0.32819722 0.10892196 0.01444619 0.02504835
#  0.01715777 0.01222231 0.4067378  0.0523994 ]
# acc :  0.41502320521898084

#정리후
# [0.38222511 0.47375412 0.06540497 0.02413377 0.02160692 0.00971394
#  0.00427154 0.00856714 0.00649685 0.00382563]
# acc col정리 :  0.8679631482172111

#model1 = GradientBoostingRegressor(max_depth=4)
#정리전
# [0.07378113 0.01309454 0.22903958 0.11018153 0.03224211 0.06459372
#  0.0323457  0.02632961 0.35591914 0.06247293]
# acc :  0.35373091823017033

#정리후
# [0.39319594 0.22452784 0.10333879 0.05565111 0.06917675 0.05265098
#  0.05238476 0.04907383]
# acc col정리 :  0.46617038120420096