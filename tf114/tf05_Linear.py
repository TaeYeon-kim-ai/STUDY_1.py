'''
y = wx + b
#참고 : 모두의딥러닝
'''

import tensorflow as tf
tf.set_random_seed(66)

x_train = [1, 2, 3]
y_train = [3, 5, 7]

W = tf.Variable(1.0)#tf.random_normal([1]), name = 'weight') #정규분포에 의한 랜덤한값 한개 넣기
b = tf.Variable(0.0)#tf.random_normal([1]), name = 'bias') #정규분포에 의한 랜덤한값 한개 넣기

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b))

#가설(모델)
hypothesis = W * x_train + b # y = wx + b 

#loss     감소 _ 평균     지승     예측값        결과값 // 컴파일
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #mse(Mean Squared Error) F분포

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) #경사하강법 최적화
train = optimizer.minimize(cost) #cost(loss) 최소화 # 최적의 loss구하기
 
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #값초기화

#train_실행시킴 --- optimizer실행 ----- (최소값)cost불러오기 ------- reduce_mean_square (mse(예측값 - 목표값) ------ hypothesis : W * x_train + b
for step in range(200) : #epochs
    sess.run(train)
    if step % 1 == 0 : #20번 스탭마다 찍어라
        print(step, sess.run(cost), sess.run(W), sess.run(b))  # W // 2, b // 1
              #에포      로스         가중치        바이어스 출력   
              # 2000 1.0781078e-05 [1.9961864] [1.0086691]         

#연산될때마다 텐서머신 내 값이 갱신됨

#[과제] epochs 3 까지 손으로 계산하기