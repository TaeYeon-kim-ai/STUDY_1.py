import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
import autokeras as ak
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)#(404, 13) (404,)
print(x_test.shape, y_test.shape)#(102, 13) (102,)

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=3
)

model.fit(x_train, y_train, epochs = 10, validation_split = 0.2)

results = model.evaluate(x_test, y_test)

model2 = model.export_model()
best_model = model.tuner.get_best_model()
best_model2.save('C:/data/h5/best_boston.h5')

# best_model = load_model('C:/data/h5/best_boston.h5')
# results = best_model.evaluate(x_test, y_test)
# print('results: ', results)
# best_model.summary()