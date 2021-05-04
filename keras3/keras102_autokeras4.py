import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(y_train[:3])

model = ak.ImageRegressor(
    overwrite=True,
    max_trials=3,
    loss = 'mae', 
    metrics=['mse'],
)

#model.summary() #안먹힘

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=3, factor=0.2, verbose=2)
ck = ModelCheckpoint('C:/data/MC/', save_weight_only = True, save_best_only = True, monitor = 'val_loss', verbose = 1)
model.fit(x_train, y_train, epochs = 15, validation_split = 0.2, callbacks = [es, lr, ck])


results = model.evaluate(x_test, y_test)
print(results)

predict = model.predict(x_test)
print(predict)

'''
Hyperparameter    |Value             |Best Value So Far
image_block_1/n...|False             |False
image_block_1/a...|True              |False
image_block_1/b...|resnet            |resnet
image_block_1/r...|False             |False
image_block_1/r...|resnet101         |resnet50
image_block_1/r...|False             |False
regression_head...|0                 |0
optimizer         |adam              |adam
learning_rate     |0.001             |0.001
'''

