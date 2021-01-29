# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
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
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

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
model1 = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model1.fit(x1_train, y1_train)

#4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)
print(model1.feature_importances_)
print("acc col정리 : ", acc1)

#정리전
# [0.         0.0624678  0.         0.         0.         0.
#  0.         0.         0.         0.02364429 0.         0.
#  0.         0.01297421 0.         0.         0.         0.
#  0.         0.         0.         0.01695087 0.         0.75156772
#  0.00738884 0.         0.00485651 0.11522388 0.         0.00492589]
# acc :  0.9210526315789473

#정리후
# [23, 27, 1, 9, 21, 13, 24, 29, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0.73861347 0.13892236 0.03908945 0.01637733 0.04313287 0.00482393
#  0.         0.         0.01904059 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.        ]
# acc col정리 :  0.9385964912280702