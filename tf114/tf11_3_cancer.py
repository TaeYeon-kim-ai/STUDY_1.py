# 이진 분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)

print(x_data.shape, y_data.shape) #(569, 30) (569,1)

x = tf.placeholder(tf.float32, shape = [None, 30])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([30, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state = 64)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#hypothesis
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) 

#COMPILE
#loss
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.0007).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(step, '\t loss', sess.run(cost, feed_dict={x:x_train, y:y_train}))

    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    print('score :', accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))

# Acc : 0.9649123
# score : 0.9649122807017544