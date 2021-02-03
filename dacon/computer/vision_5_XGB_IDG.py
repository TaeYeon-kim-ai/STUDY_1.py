import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#DATA
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
test = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

x = train.drop(['id', 'digit', 'letter'], axis=1).values
x_pred = test.drop(['id', 'letter'], axis = 1).values

x = x.reshape(-1, 28, 28, 1)
x = x/255
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255

y = train['digit']

y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, stratify = y)

#ImageDataGenerator
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

train_generator = idg.flow(x_train,y_train,batch_size=8)
test_generator = idg2.flow(x_test,y_test)
pred_generator = idg2.flow(x_pred,shuffle=False)

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
learning_hist = model.fit(x_train,y_train, verbose = 1, 
        eval_metric = ['merror', 'mlogloss'], 
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds= 50)


#3.1 시각화
hist = pd.DataFrame(learning_hist.history)
hist['val_loss'].min

hist.columns
plt.title('Training and validation loss')
plt.xlabel('epochs')

plt.plot(hist['val_loss'])
plt.plot(hist['loss'])
plt.legend(['val_loss', 'loss'])

plt.figure()

plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Traning and validation accuracy')

plt.show()

#평가
result = model.score(test_generator)
y_pred = model.predict(pred_generator)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
results = model.evals_result()

#test1
submission = pd.read_csv('C:/STUDY/dacon/computer/submission.csv')
submission['digit'] = result.argmax(1)
submission.to_csv('C:/STUDY/dacon/computer/2021.02.02.csv',index=False)


