import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

#2. MODEL
w = tf.Variable(tf.random_normal([784, 100]), name = 'weight')
b = tf.Variable(tf.random_normal([100]), name = 'bias')
# layer1 = tf.nn.softmax(tf.matmul(x, w) + b)
# layer1 = tf.nn.relu(tf.matmul(x, w) + b)
# layer1 = tf.nn.selu(tf.matmul(x, w) + b)
layer1 = tf.nn.elu(tf.matmul(x, w) + b)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)# 30% dropout시키겠다.

w2 = tf.Variable(tf.random_normal([100, 50], name='weight2'))
b2 = tf.Variable(tf.random_normal([50], name='bias2'))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)# 30% dropout시키겠다.

w3 = tf.Variable(tf.random_normal([50, 10]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([10]), name = 'bias3')
hypothesis = tf.nn.relu(tf.matmul(layer2, w3) + b3)

#binary_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001) :
        _, cost_val = sess.run([optimizer, cost], feed_dict={x : x_train, y: y_train})

        if step % 200 == 0 :
            print(step, "[loss] : ", cost_val)

    a = sess.run(hypothesis, feed_dict={x : x_test})
    print("acc : ", accuracy_score(sess.run(tf.argmax(y_test, 1)), sess.run(tf.argmax(a, 1))))
