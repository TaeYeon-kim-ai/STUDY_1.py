import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

#sigmoid #레이어의 구성을 행렬 곱에 맞게 연산을 해준다.
x = tf.placeholder(tf.float32, shape=[None, 2]) #(N,2)
y = tf.placeholder(tf.float32, shape=[None, 1]) #(2, 10)

#hidden layer1 #히든레이어의 노드는 마음대로 할 수 있다.
w1 = tf.Variable(tf.random_normal([2, 10]), name = 'weigth1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dese(10, input_dim = 2, activation = 'sigmoid'))

#hidden layer2
w2 = tf.Variable(tf.random_normal([10, 7]), name = 'weigth2')
b2 = tf.Variable(tf.random_normal([7]), name = 'bias2')
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)
# model.add(Dese(7, activation = 'sigmoid'))

#output 
w3 = tf.Variable(tf.random_normal([7, 1]), name = 'weigth2')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)
# model.add(Dese(1, activation = 'sigmoid')) #bias

#binary_crossentropy
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) #binary_crossentropy

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

# 예측값 :  [[0.03026746]
#  [0.9578239 ]
#  [0.96481943]
#  [0.04673885]]
#  원래값 :
#  [[0.]
#  [1.]
#  [1.]
#  [0.]] Accuracy :  1.0