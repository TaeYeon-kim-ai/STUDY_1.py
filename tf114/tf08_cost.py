import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.] #그냥 파이썬의 veriable
y = [2., 4., 6.] #그냥 파이썬의 veriable

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y)) #mse

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess :
    #sess.run(tf.compat.v1.global_variables_initializer()) #안해줘도 돌아감
    #텐서플로의 실질적인 w에 tf.valiable설정은 안해줘서 파이썬의 veriable로 취급하는 듯
    for i in range(-30, 50) : #-30 ~ 50
        curr_w = i * 0.1 #i*0.1단위로 증가 -3, 
        curr_cost = sess.run(cost, feed_dict={w : curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("=========================================")
print("W : ", w_history)
print("=========================================")
print("cost : ", cost_history)
print("=========================================")

plt.plot(w_history, cost_history)
plt.show()