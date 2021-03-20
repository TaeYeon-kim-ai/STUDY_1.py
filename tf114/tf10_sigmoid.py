import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1, 2], [2,3], [3,1],
          [4,3], [5,3], [6,2]] #(6,2)

y_data = [[0], [0], [0],
          [1], [1], [1]]#(6,1)


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([2,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) #sigmoid적용 #결과값 activation으로 쌋다.

#cost = tf.reduce_mean(tf.square(hypothesis - y))
#binary_crossentropy
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) 

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

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
