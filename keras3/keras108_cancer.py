import numpy as np
import autokeras as ak
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

x_train, x_test, y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, train_size = 0.8, random_state = 77)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = ak.StructuredDataRegressor(overwrite = True,
                                   max_trials = 1,
                                   loss = 'mse',
                                   metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=6)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factr = 0.5, verbose = 2)
model.fit(x_train, y_train, epochs = 1, callbacks = [es, lr], validation_split = 0.2)

# SAVE Best Model
# model = model.export_model()
best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/best_cancer.h5')

# LOAD Best Model
best_model = load_model('C:/data/h5/best_cancer.h5')
results = best_model.evaluate(x_test, y_test)
print('results: ', results)
best_model.summary()