import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

#'''
#print('hypotersis : ', ???)

# [실습]
#1. sess.run()
#2. InteractiveSession()
#3. .eval(session=sess)
# hypothesis를 출력하는 코드를 만드시오.

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# aaa = sess.run(hypothesis)
# print("aaa.hypotersis : ", aaa) #aaa :  [2.2086694]
# sess.close() 

#다른 방법
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc.hypotersis : ", ccc)
sess.close()

#다른 방법 InteractiveSession
#WARNING제거 
sess = tf.compat.v1.InteractiveSession() #아래있는것 다 sess불러와짐 #tf.Tensor.eval 이나 tf.Operation.run할 때 sess된걸로 침
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval() #
print("bbb.hypotersis : ", bbb) #bbb :  [2.2086694]
sess.close()


