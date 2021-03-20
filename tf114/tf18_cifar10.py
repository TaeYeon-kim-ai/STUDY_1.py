import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

print(tf.__version__)

#1. DATA
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 10) (10000, 10)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')/255

learning_rate = 0.0001
training_epochs = 10
batch_size = 32
total_batch = int(len(x_train)/batch_size) #60000/100
drop = 0.001

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [10])

#2. MODLE

#L1
w1 = tf.compat.v1.get_variable("w1", shape= [3, 3, 3, 32]) # 3, 3 = 커널 사이즈, 1 = channel(input_dim), 32 = filter
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1],padding="SAME")
print(L1)

w2 = tf.compat.v1.get_variable("w2", shape= [3, 3, 32, 128]) # 3, 3 = 커널 사이즈, 1 = channel(input_dim), 32 = filter
L2 = tf.nn.conv2d(L1, w2, strides = [1,1,1,1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1],padding="SAME")
print(L2)#shape=(None, 8, 8, 128), dtype=float32)

w3 = tf.compat.v1.get_variable("w3", shape= [3, 3, 128, 64]) # 3, 3 = 커널 사이즈, 1 = channel(input_dim), 32 = filter
L3 = tf.nn.conv2d(L2, w3, strides = [1,1,1,1], padding="SAME")
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1],padding="SAME")
print(L3)#shape=(None, 4, 4, 64), dtype=float32)

Lflat = tf.reshape(L3, [-1, 4*4*64])

w4 = tf.compat.v1.get_variable("w4", shape = [4*4*64, 64])
b4 = tf.Variable(tf.compat.v1.zeros([64]), name = 'b4')
L4 = tf.nn.relu(tf.matmul(Lflat, w4) + b4)
L4 = tf.compat.v1.nn.dropout(L4, keep_prob=drop)
print(L4)#shape=(None, 64)

w5 = tf.compat.v1.get_variable("w5", shape = [64, 10])
b5 = tf.Variable(tf.compat.v1.zeros([10]), name = 'b5')
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)
print("최종 : ",hypothesis) #shape=(None, 10)

#COMPLIE
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) #categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#batch...
for epoch in range(training_epochs) : 
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

print("End")

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict={x : x_test, y : y_test}))


