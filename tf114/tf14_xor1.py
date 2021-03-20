import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

#sigmoid
x = tf.placeholder(tf.float32, shape=[None, 2])
hidden = tf.placeholder(tf.float32, shape=[None, 5])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2,1]), name = 'weigth1')
w2 = tf.Variable(tf.random_normal([5,1]), name = 'weigth2')
b1 = tf.Variable(tf.random_normal([1]), name = 'bias1')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')

hypothesis1 = tf.nn.relu(tf.matmul(x, w1) + b1) # (N, 4) x (4, 3) + (1, 3) , ... (N , 3) + (1, 3) = ..
hypothesis = tf.nn.sigmoid(tf.matmul(hypothesis1, w2) + b2)

#binary_crossentropy
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) 

train = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32)) #equal했을때 같으면 1 틀리면 0반환

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001) :
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})

    if step % 200 == 0 :
        print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x : x_data, y : y_data})
    print("예측값 : ", h, "\n", "원래값 : ", "\n",c, "Accuracy : ", a)
