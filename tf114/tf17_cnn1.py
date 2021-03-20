import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1.DATA
from tensorflow.keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255

learning_rate = 0.0001
training_epochs = 15
batch_size = 32
total_batch = int(len(x_train)/batch_size) #60000/100
drop = 0.001

x = tf.placeholder(tf.float32, [None, 28, 28, 1]) #shape그대로 넣기
y = tf.placeholder(tf.float32, [None, 10])

#2. MODEL

#L1
#                                kernel_size, cannel(color(3)/ gray(1)), filter(output):다음레이어 전달 node
w1 = tf.get_variable("w1", shape = [3, 3, 1, 32])
#                                 D4 맞추기   
L1 = tf.nn.conv2d(x, w1, strides= [1,1,1,1], padding='SAME') #==============  28, 28, 32로 전달  .. padding "SAME" 으,로 동일
print(L1)
# Conv2D(filter, kernel_size, input_shape) #서머리
# Conv2D(32, (3,2), input_shape = 28,28,1) #파라미터 갯수 ====커널사이즈에 input dim 곱하기
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L1) # shape=(?, 14, 14, 32)

#L2
w2 = tf.get_variable("w2", shape = [3, 3, 32, 64]) #
L2 = tf.nn.conv2d(L1, w2, strides= [1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L2) # shape=(?, 7, 7, 64)

#L3
w3 = tf.get_variable("w3", shape = [3, 3, 64, 128])
L3 = tf.nn.conv2d(L2, w3, strides= [1,1,1,1], padding='SAME') #7, 7, 128
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L3) #shape=(?, 4, 4, 128)

#L4
w4 = tf.get_variable("w4", shape = [3, 3, 128, 64])
L4 = tf.nn.conv2d(L3, w4, strides= [1,1,1,1], padding='SAME') #7, 7, 128
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') # 7, 7, 32
print(L4) #shape=(?, 2, 2, 64)

#Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64]) # D2로 줄이기... 그냥 곱하네
print("플래튼 : ", L_flat) #shape=(?, 256)

w5 = tf.get_variable("w5", shape = [2*2*64, 64], initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)) # 위에
b5 = tf.Variable(tf.zeros([64]), name = 'b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
#L5 = tf.nn.dropout(L5, keep_prob=drop)
print(L5) # shape=(?, 64)

w6 = tf.get_variable("w6", shape = [64, 32], initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)) # 위에
b6 = tf.Variable(tf.zeros([32]), name = 'b6')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
#L6 = tf.nn.dropout(L6, keep_prob=drop) 
print(L6) # shape=(?, 32)

w7 = tf.get_variable("w7", shape = [32, 10], initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)) # 위에
b7 = tf.Variable(tf.zeros([10]), name = 'b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print("최종출력 : ", hypothesis) # shape=(?, 10)

# 컴파일, 훈련
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) #categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#COMPILE
sess = tf.Session()
sess.run(tf.global_variables_initializer())

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

print("훈련 끗!!!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict={x : x_test, y : y_test}))


#Acc :  0.7453 # epochs = 15 // drop 0.02 #learning_rate = 0.0005
#Acc :  0.9902 # epochs = 15 // drop 삭제 #learning_rate = 0.0001 #epoch :  0015 cost = 0.004698698






