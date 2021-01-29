from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_iris()
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