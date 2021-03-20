# 회귀
from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target.reshape(-1, 1)
print(x_data.shape, y_data.shape) # (442, 10) (442, 1)

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state= 64)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#hypothesis
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate = 0.55).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict = {x : x_train, y : y_train})

    if step % 200 == 0 :
        print(step, cost_val, hy_val)

#최종 sklearn의 r2_score값으로 출력할것
score = r2_score(y_train, hy_val)
print("r2_score", score) #r2_score 0.520805870682859










