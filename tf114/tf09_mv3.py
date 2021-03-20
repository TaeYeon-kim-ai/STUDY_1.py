import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

dataset = np.loadtxt("C:/data/csv/data-01-test-score.csv", delimiter=',') #???
print(dataset.shape)
#[실습] 만드러보자
x_train = dataset[:,:-1]
y_train = dataset[:,-1:]

x_pred = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]] #(5,3) metrix

#y_train = dataset[:, -1].reshape(-1, 1)
print(x_train.shape)#25,3
print(y_train.shape)#25,1

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

7#행 맞춰두고 열은 y의 열값 x*w를 한 shape와 y의 shape같게
w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b #matmul 은 매트릭스 멀티(매트릭스 곱)

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00003)
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) : 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict = {x : x_train, y : y_train})

    if step % 100 == 0 :
        print("step : ", step, "\n", "cost : ", cost_val, "\n", hy_val)
print("===================================================") 
print("x_pred", "\n", sess.run(hypothesis, feed_dict = {x : x_pred}))

sess.close()

#predict
#73, 80, 75, 152
#93, 88, 93, 185
#89, 91, 90, 180
#96, 98, 100, 196
#73, 66, 70, 142
