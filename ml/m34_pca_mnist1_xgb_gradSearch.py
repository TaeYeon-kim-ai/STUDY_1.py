#m31로 만든 0.95 이상의 n_component = ?를 사용하여
# GridSearch _XGB 모델을 만들 것

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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
#print("cumsum : ", cumsum)

d = np.argmax(cumsum >= 0.95)+1 #95 가능범위 확인
#print("cumsum >= 0.99", cumsum >= 0.99)
print("d : ", d)

pca = PCA(n_components= d, ) #차원축소
x2 = pca.fit_transform(x)
print(x2.shape)

# #1.1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size = 0.8, random_state = 55)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (70000, 154)
# (44800, 154) (14000, 154) (11200, 154)
# (44800,) (14000,) (11200,)


parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]
kfold = KFold(n_splits=4, shuffle= True)

#2. 모델링
model = GridSearchCV(XGBClassifier(n_job = -1), parameters, cv = kfold) #use_label_encoder= False


#3. 컴파일, 훈련
model.fit(x_train, y_train, eval_metric = 'mlogloss', verbose = True, eval_set = [(x_train, y_train), (x_test, y_test)])#evaluate


#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc col정리 : ", acc)

y_pred = model.predict(x_test[:10])
print("acc : ", accuracy_score(y_test, y_pred))

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
