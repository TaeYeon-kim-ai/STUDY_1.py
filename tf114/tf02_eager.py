#from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) #False // True

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) #False // False

print(tf.__version__) #2.4.0 // 1.14.0

hello = tf.constant("Hello World") #constant = 상수
print(hello)

# Tensor("Const:0", shape=(), dtype=string) hellow world안뜸
# 따라서 텐서플로1.대에서는 session을 만들고 실행시켜야 출력된다 
# 이걸 해결하기 위해 파이토치가 나왔다.
# 파이토치로 넘어간 사람도 있다.
sess = tf.compat.v1.Session()
print(sess.run(hello)) #b'Hello World

#tensorflow 3가지 자료형 
#constant // Placeholder // Variabl
