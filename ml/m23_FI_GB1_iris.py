# 컬럼 drop
#피쳐처임포턴스가 전체 중요도에서 25%미만인 컬럼들을 제거

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# x = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# x = x.iloc[:, [1, 7, 10, 17, 20, 21, 22, 24, 26, 27]]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

print(len(dataset['feature_names']))
df=pd.DataFrame(dataset.data,columns =[dataset.feature_names])

#2. 모델
model = DecisionTreeClassifier()
model = RandomForestClassifier()
model = GradientBoostingRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

#DecisionTree
# [0.0125026  0.         0.03213177 0.95536562]
# acc :  0.9333333333333333

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


