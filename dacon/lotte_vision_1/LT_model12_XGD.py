import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.applications import AlexClassifier


#1.DATA
x = np.load('../../data/npy/train_x_192.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_192.npy', allow_pickle=True)
target = np.load('../../data/npy/predict_x_192.npy', allow_pickle=True)

#파라미터튜닝 적용
parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01], "max_depth" : [4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

# polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 66 , shuffle = True)

kfold = KFold(n_splits= 5, shuffle= True)

model = RandomizedSearchCV(XGBClassifier(), parameters, cv = kfold)

score = cross_val_score(model, x_train ,y_train, cv= kfold)

print("최적의 매개변수 : ", model.best_estimator_)

acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)