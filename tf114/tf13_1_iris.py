from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y_data = y_data.reshape(-1,1)                 #. y_train => 2D
one.fit(y_data)                          #. Set
y_data = one.transform(y_data).toarray()      #. transform
print(x_data.shape, y_data.shape) #(150, 4) (150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state = 64)

x = tf.placeholder('float', [None, 4]) # 150, 4
y = tf.placeholder('float', [None, 3]) # 150, 3

w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x : x_train, y: y_train})

        if step % 20 == 0 :
            print(step, "[loss] : ", cost_val)
    
    a = sess.run(hypothesis, feed_dict = {x : x_test})
    print("acc : ", accuracy_score(sess.run(tf.argmax(y_test, 1)), sess.run(tf.argmax(a, 1)))) #acc :  0.9