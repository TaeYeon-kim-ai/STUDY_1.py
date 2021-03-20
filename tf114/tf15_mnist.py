#[실습]  mnist학습
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
y_train = y_train.reshape(-1, 1) # (60000, 1)
y_test = y_test.reshape(-1, 1)

x_train = x_train.reshape(-1, 28*28) # (60000, 784)
x_test = x_test.reshape(-1, 28*28) # (10000, 784)
print(x_train.shape, x_test.shape)

from  sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

w1 = tf.Variable(tf.zeros([784, 64]), name = 'weight1')
b1 = tf.Variable(tf.zeros([64]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.zeros([64, 32]), name = 'weight2')
b2 = tf.Variable(tf.zeros([32]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.zeros([32, 10]), name = 'weight3')
b3 = tf.Variable(tf.zeros([10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(100) :
        _, cost_val = sess.run([optimizer, cost], feed_dict={x : x_train, y: y_train})
        
        if step % 1 == 0 :
            print(step, "[loss] : ", cost_val)
        
    #predict
    a = sess.run(hypothesis, feed_dict={x : x_test})
    print("acc : ", accuracy_score(sess.run(tf.argmax(y_test, 1)), sess.run(tf.argmax(a, 1))))

    