import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

print(x_train.shape, x_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_train)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2,
    loss = 'loss', 
    metrics=['acc'],
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
es = EarlyStopping(monitor='val_;oss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=3, factor=0.2, verbose=2)
ck = ModelCheckpoint('C:/data/MC/', save_weight_only = True, save_best_only = True, monitor = 'val_loss', verbose = 1)
model.fit(x_train, y_train, epochs = 3, validation_split = 0.2, callbacks = [es, lr, ck])

results = model.evaluate(x_test, y_test)

print(results)