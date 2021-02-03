import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#DATA
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
x_pred = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

print(train.shape) # (2048, 787)
print(x_pred.shape) # (20480, 786)

x = train.drop(['id', 'digit', 'letter'], axis=1).values
x = x/255

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

# d = np.argmax(cumsum >= 0.95)+1 #95 가능범위 확인
# #print("cumsum >= 0.99", cumsum >= 0.99)
# print("d : ", d)

# pca = PCA(n_components= d, ) #차원축소
# x2 = pca.fit_transform(x)
# print(x2.shape)


y = train['digit']

y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False, random_state = 0)


#MODELing
kfold = KFold(n_splits = 5, shuffle = True)
parameters = [
    {"n_estimators" : [200], 
    "learning_rate " : [0.01],
    "max_depth" : [64],
    "colsample_bytree" : [0.9],
    "colsample_bylevel" : [0.9]}
    ]

model = XGBClassifier(parameters)

#PRAC
model.fit(x_train,y_train, verbose = 1, 
        eval_metric = ['merror', 'mlogloss'], 
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds= 50)


#평가
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
results = model.evals_result()

#acc :  0.4878048780487805

'''
cnn_test = x_pred.drop(['id', 'letter'], axis=1).values
cnn_test = cnn_test.reshape(-1, 28, 28, 1)
cnn_test = cnn_test/255

submission = pd.read_csv('C:/STUDY/dacon/computer/submission.csv')
submission['digit'] = np.argmax(model.predict(cnn_test), axis=1)
submission.head()

submission.to_csv('C:/STUDY/dacon/computer/2021.02.02_1.csv', index=False)


#시각

epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label = 'train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label = 'test')
ax.legend()
plt.ylabel('mlog Loss')
plt.title('XGBoost mLog Loss')
#plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label = 'train')
ax.plot(x_axis, results['validation_1']['merror'], label = 'test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost MERROR')
plt.show()
'''