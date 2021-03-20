#레이어 구성 / 이니설라이즈 구성

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
#w1 = tf.Variable(tf.random_normal([784, 100]), name = 'weight1')
w1 = tf.get_variable('weight1', shape=[784, 100],
                    initializer= tf.contrib.layers.xavier_initializer()) #xavier == kannel initialize
print("w1 : ", w1) #tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>
b1 = tf.Variable(tf.random_normal([100]), name = 'bias1')
print("b1 : ", b1)
layer1 = tf.nn.elu(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.softmax(tf.matmul(x, w) + b)
# layer1 = tf.nn.relu(tf.matmul(x, w) + b)
# layer1 = tf.nn.selu(tf.matmul(x, w) + b)
print("layer1 : ", layer1)
layer1 = tf.nn.dropout(layer1, keep_prob=0.3)# 30% dropout시키겠다.
print("layer1 : ", layer1)

w2 = tf.get_variable('weight2', shape = [100, 128], 
                    initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.Variable(tf.random_normal([128], name='bias2'))
layer2 = tf.nn.elu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)# 30% dropout시키겠다.

w3 = tf.get_variable('weight3', shape = [128, 64], 
                    initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([64]), name = 'bias3')
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.3)# 30% dropout시키겠다.

w4 = tf.get_variable('weight4', shape = [64, 10], 
                    initializer=tf.contrib.layers.xavier_initializer()) 
b4 = tf.Variable(tf.random_normal([10]), name = 'bias4')
hypothesis = tf.nn.relu(tf.matmul(layer3, w4) + b4)



#binary_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.006).minimize(cost)

'''
#===============batch_size
'''
trining_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#배치정의
for epoch in range(trining_epochs) : 
    avg_cost = 0
    for i in range(total_batch) : #600번 돈다
        start = i * batch_size  # 0 ~
        end = start + batch_size # 0~ 100
        batch_x, batch_y = x_train[start : end], y_train[start : end]  #i = 0 = 0 ~ 100까지의 데이터 불러와서       
        feed_dict = {x : batch_x, y : batch_y}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict) #c에 cost출력
        avg_cost += c/total_batch # += c와 total batch한거를 acg_cost에 넣겠다.  cost / total_batch(600) 나눈걸 avg_cost에 저장 1epoch 에 600개 다 더한다음 600으로 나누면 평균로스

    print('epoch : ', '%04d' %(epoch + 1), 
          'cost = {:.9f}'.format(avg_cost)
    )

print("훈련 끗!!!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict={x : x_test, y : y_test}))
'''
총 데이터 60000개 나는 100개를 훈련시키는데, 1에 100에 대한 loss가 나오고 이를 600개 하면 600개에 대한 loss가 나온다 이를 더해서 다시 600으로 나누면 평균 loss값이 나온다. 
값은 차이가 날 수 있겠지만 유사하니, 별 차이 없다고 보면 된다.(컴퓨터 안터지게 하려고)/ ====/ keras의 model.fit에 있는 batch_size 라 할 수 있음.
'''

# with tf.Session() as sess :
#     sess.run(tf.global_variables_initializer())
    
#     for step in range(2001) :
#         _, cost_val = sess.run([optimizer, cost], feed_dict={x : x_train, y: y_train})

#         if step % 200 == 0 :
#             print(step, "[loss] : ", cost_val)

#     a = sess.run(hypothesis, feed_dict={x : x_test})
#     print("acc : ", accuracy_score(sess.run(tf.argmax(y_test, 1)), sess.run(tf.argmax(a, 1))))
