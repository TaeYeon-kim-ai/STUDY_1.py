from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)


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

# [0.         0.00497399 0.         0.0564906  0.         0.
#  0.1594946  0.         0.         0.07239443 0.         0.33754988
#  0.36909649]
# acc :  0.9166666666666666
