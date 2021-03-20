import tensorflow as tf
tf.set_random_seed(66)

#컬럼 2개 이상일 경우 
# y = w1x1 + w2x2 + w3x3 + b

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 9., 100., 70.]
y_data = [152., 185., 180., 196., 142.]


#입력할 placegolder
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict = {x1 : x1_data, x2: x2_data, x3: x3_data, y: y_data})
    
    if step % 200 == 0 :
        print(step, "cost : ", cost_val, "\n", hy_val)

sess.close()

#nan 정상적으로 나오게 출력
# 0 cost :  15257.463 
 #[46.07737  43.7147   69.432106 52.270447 31.443575]    

