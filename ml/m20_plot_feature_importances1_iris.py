from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

#0. 함수정의
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
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = get_column_index(model)
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

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

#DecisionTree
# [0.0125026  0.         0.03213177 0.95536562]
# acc :  0.9333333333333333
