import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

print(x_train.shape, x_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_train)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)


model.fit(x_train, y_train, epochs = 3)

results = model.evaluate(x_test, y_test)

print(results)

'''
Trial 1 Complete [00h 01m 18s]
val_loss: 0.04152538254857063

Best val_loss So Far: 0.04152538254857063
Total elapsed time: 00h 01m 18s

Search: Running Trial #2

Hyperparameter    |Value             |Best Value So Far
image_block_1/b...|resnet            |vanilla
image_block_1/n...|True              |True
image_block_1/a...|True              |False
image_block_1/i...|True              |None
image_block_1/i...|True              |None
image_block_1/i...|0                 |None
image_block_1/i...|0                 |None
image_block_1/i...|0.1               |None
image_block_1/i...|0                 |None
image_block_1/r...|False             |None
image_block_1/r...|resnet50          |None
image_block_1/r...|True              |None
classification_...|global_avg        |flatten
classification_...|0                 |0.5
optimizer         |adam              |adam
learning_rate     |0.001             |0.001
'''


