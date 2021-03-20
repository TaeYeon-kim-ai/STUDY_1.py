#가중치 불러오기


import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')

print(W) #<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

#세션에 변수를 선언할 때 마다  session.run이 있고
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print("aaa : ", aaa) #aaa :  [2.2086694]
sess.close() 

#다른 방법 InteractiveSession
#WARNING제거 
#sess = tf.InteractiveSession() 
#sess.run(tf.global_variables_initializer())
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval() #변수.eval()
print("bbb : ", bbb) #bbb :  [2.2086694]
sess.close()

#다른 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc : ", ccc) #ccc :  [2.2086694]
sess.close()
