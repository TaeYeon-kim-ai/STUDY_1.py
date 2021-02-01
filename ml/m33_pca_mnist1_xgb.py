#m31로 만든 0.95 이상의 n_component = ?를 사용하여
#XGB 모델을 만들 것

#mnist dnn 보다 성능 좋게 만들어라!!
#cnn과 비교!!
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape) #(70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])/255.

#MinMax
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
#print("cumsum : ", cumsum)

d = np.argmax(cumsum >= 0.95)+1 #95 가능범위 확인
#print("cumsum >= 0.99", cumsum >= 0.99)
print("d : ", d)

#시각화
# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components= d , ) #차원축소
x2 = pca.fit_transform(x)
print(x2.shape)
# d :  331
#(70000, 331)

#1.1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size = 0.8, random_state = 55)
print(x_train.shape, x_test.shape) #(56000, 154) (14000, 154)
print(y_train.shape, y_test.shape) #(56000,) (14000,)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=55)

# print(x_train.max)
# print(x_train.min)

#2. 모델링
model = XGBClassifier(n_job = -1, use_label_encoder= False)

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
# print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

#DNN
#(784, )
# loss :  [0.09116600453853607, 0.9779000282287598]
# [7 2 1 0 4 1 4 9 5 9]

#PCA 154
# loss :  [0.13378241658210754, 0.9748571515083313]
# [9 4 5 3 8 8 8 1 6 4]

#PCA 331
