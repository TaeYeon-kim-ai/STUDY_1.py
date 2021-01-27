#딥러닝 모델로 XOR문제 해결

from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
#XOR형(분류) 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]
#  0  1
#  1  0

#2. 모델(머신러닝)
# model = LinearSVC() 
model = Sequential()
model.add(Dense(1, input_dim = 2))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# y_pred :  [[4.7916174e-04]
#  [9.9982142e-01]
#  [9.9980938e-01]
#  [4.6679378e-04]]
# 1/1 [==============================] - 0s 0s/step - loss: 3.2884e-04 - acc: 1.0000
# model.score :  1.0

#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_data, y_data, batch_size= 1, epochs=100)


#4. 평가, 예측
y_pred = model.predict(x_data)
print("y_pred : ", y_pred) # x_data에 대한 예측값 : y_pred :  [0 0 0 1]

result = model.evaluate(x_data, y_data) 
print("model.score : ", result[1]) #[0]은 loss 출력 [1] acc출력

#accuracy_score
acc = accuracy_score(y_data, y_pred)
#                   (실데이터, 예측결과 데이터)
print("accuracy_score : ", acc)

#SVC() 모델 사용
# y_pred :  [0 1 1 0]
# model.score :  1.0
# accuracy_score :  1.0