# [텐서기계]placehoder 로 들어가고 variable로 연산하고 sess.run을 통해 출력한다.
# [실습] placeholder사용
'''
y = wx + b
#참고 : 모두의딥러닝
'''

import tensorflow as tf
tf.set_random_seed(66)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(1.0)#tf.random_normal([1]), name = 'weight') #정규분포에 의한 랜덤한값 한개 넣기
b = tf.Variable(0.0)#tf.random_normal([1]), name = 'bias') #정규분포에 의한 랜덤한값 한개 넣기

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b))

#가설(모델)
hypothesis = W * x_train + b # y = wx + b 

#CAMPILE
#loss     감소 _ 평균     지승     예측값        결과값 // 컴파일
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #mse(Mean Squared Error) F분포

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) #경사하강법 최적화
# train = optimizer.minimize(cost) #cost(loss) 최소화 # 최적의 loss구하기
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) 

#feed_dict
fd = {x_train : [1,2,3], y_train : [3, 5, 7]} #트리구조에 한방에 몰아넣어서 돌리기 <=내가한거

# with문으로 구성 === sess.close() 대체
with tf.Session() as sess: #tf.Session() 을 sess로 정의
    sess.run(tf.global_variables_initializer())

    for step in range(200) : #epochs
        #sess.run(train, fd)
        #sess.run에 대한건 모든게 반환 가능하다
        # 반환 train, cost, W, b *train은 값을 반환할 필요가 겂으니 '_'로 해둔다
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train : [1,2,3], y_train : [3,5,7]})
        if step % 1 == 0 : #20번 스탭마다 찍어라
            #print(step, sess.run(cost), sess.run(W), sess.run(b))  # W // 2, b // 1
            #print(step, sess.run(cost, fd), sess.run(W), sess.run(b)) <==내가한거
            print(step, cost_val, W_val, b_val)



#tensor 기계

#placeholder<----------------------------.
#=============================           .
#                                        .
        #variable                        .
#                                        .
#                                        .
#=============================   .... sess













