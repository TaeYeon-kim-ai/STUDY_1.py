#과제
#       차원    형태    내용    
#스칼라         1
#벡터           [1,2,3]
#행렬           [[1,2],[2,3]]
#텐서           [[[1,2],[1,2,3]]]

# x * W 두개의 사이즈가 맞아야함 
# x = 5, 3
# W = 3, 1(2,3,4,되던 상관없음) + b와 더할 수 있는 shape가 동일해야함.
# (5, 3) x (3, 1) = (5, 1) #앞에 열과 뒤에 행만 맞으면 행렬 연산할 수 있음
# (3, 2) x (2, 3) = (3, 3)
# [실습] 만들어봐
# verbose 로 나오는건 step과 cost와 hypothesis  // epochs = 2001, 10개단위
import tensorflow as tf 
tf.set_random_seed(66)

x_data = [[73, 51, 65],
          [92, 98, 40],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]] #(5,3) metrix

y_data = [[152],
          [185],
          [180],
          [205],
          [142]] #metrix(5,1)

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

#행 맞춰두고 열은 y의 열값 x*w를 한 shape와 y의 shape가 같아야한다.
w = tf.Variable(tf.random_normal([3, 1]), name = 'weight') 
b = tf.Variable(tf.random_normal([1]), name = 'bias') #바이어스 하나임

#hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b #matmul 은 매트릭스 멀티(매트릭스 곱)

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 71e-6) # cost :  294.25824
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.1) #cost :  176.789 [실험]
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) : 
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict = {x : x_data, y : y_data})

    if step % 20 == 0 :
        print("step : ", step, "\n", "cost : ", cost_val, "\n", hy_val)

sess.close()

'''
import matplotlib.pyplot as plt

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
'''


