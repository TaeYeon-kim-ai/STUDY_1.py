from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. 데이터
#AND형(분류) 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 0, 0, 1]
# 0  0
# 0  1

#2. 모델(머신러닝)
model = LinearSVC()


#3. 훈련
model.fit(x_data, y_data)


#4. 평가, 예측
y_pred = model.predict(x_data)
print("y_pred : ", y_pred) # x_data에 대한 예측값 : y_pred :  [0 0 0 1]

result = model.score(x_data, y_data)
print("model.score : ", result) # acc : model.score :  1.0 (100%)

#accuracy_score
acc = accuracy_score(y_data, y_pred)
#                   (실데이터, 예측결과 데이터)
print("accuracy_score : ", acc)

#result
#y_pred :  [0 0 0 1]
# model.score :  1.0
# accuracy_score :  1.0