import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델링
model = Sequential()
model.add(Dense(4, input_dim = 1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

#weigth불러오는 방법 2개 왜??
#전이학습을 위함 _ none_trainable_weights
#print(model.weights)
#print(model.trainable_weights)

print(len(model.weights))#8 각 레이어마다 (weight 1, bias 1) * 4 = 8
print(len(model.trainable_weights))#8 #각 레이어마다 (weight 1, bias 1) * 4 = 8




# [<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=array([[ 0.9267316, -0.5007937, -0.1409778, -0.8999022]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 
# 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
# array([[ 0.03968066, -0.3694688 , -0.08701241],
#        [ 0.43100333,  0.00941777, -0.7555575 ],
#        [-0.63145846,  0.74593174, -0.71733916],
#        [ 0.79102015,  0.8598515 ,  0.8236358 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.9479286 ,  0.8720801 ],
#        [-0.791986  ,  0.7495003 ],
#        [ 0.5632316 ,  0.10785413]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[ 1.4074434],
#        [-1.3753588]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]