import tensorflow as tf
print(tf.__version__) #2.4.0 // 1.14.0

hello = tf.constant("Hello World") #constant = 상수
print(hello)

#Tensor("Const:0", shape=(), dtype=string) hellow world안뜸
#따라서 텐서플로1.대에서는 session을 만들고 실행시켜야 출력된다
sess = tf.Session()
print(sess.run(hello)) #b'Hello World

#tensorflow 3가지 자료형 
#constant // Placeholder // Variable

# x = tf.compat.v1.tf.constant("Hello World")
# print(x)