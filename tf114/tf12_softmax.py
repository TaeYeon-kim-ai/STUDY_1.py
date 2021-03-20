import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = [[1,2,1,1], 
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]


x = tf.placeholder('float', [None, 4]) #(N, 4)(4, 3) = (N, 3) ... 
y = tf.placeholder('float', [None, 3])

# 행렬의 덧셈은 shape가 같아야함 행렬곱으로 맞춰줘야함
w = tf.Variable(tf.random_normal([4, 3]), name = 'weight') # ?
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')  # ?

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # (N, 4) x (4, 3) + (1, 3) , ... (N , 3) + (1, 3) = ..

#input --- output 레이어 통과하 때 activation으로 감싼다. softmax는 레이어 통과할 때 가장큰 애만 1로 반환해줌

#cost = tf.reduce_mean(tf.square(hypothesis - y))
#loss를 평균한다, 더해서 hypothesis에 로그를 취해서 가장 큰값을 1로 반환한다.
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, cost_val = sess.run([optimizer, loss], feed_dict={x : x_data, y: y_data})
        
        if step % 200 == 0 :
            print(step, "[loss] : ", cost_val)

    #predict
    a = sess.run(hypothesis, feed_dict={x : [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))#가장 높은 값에 1을 줘서 출력해줘라


    


















