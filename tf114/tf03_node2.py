# 실습
# 덧셈
# 뺼셈
# 곱셈
# 나눗셈
#만드시오

import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

sess = tf.Session()
print('sess.run(node1, noder2)', sess.run([node1, node2]))

#덧셈
node3 = tf.add(node1, node2)
print(sess.run(node3)) #7.0
#뺄셈
node4 = tf.subtract(node1, node2)
print(sess.run(node4)) #-1.0

#곱셈
node5 = tf.multiply(node1, node2)
print(sess.run(node5)) #12.0

#나눗셈
node6 = tf.divide(node1, node2)
print(sess.run(node6)) #0.75

#나눈 나머지
node7 = tf.negative(node1, node2)
print(sess.run(node7))

#제곱
node8 = tf.pow(node1, node2)
print(sess.run(node8))
'''
# 6. 나눈 나머지
mod = tf.mod(node1, node2)

# 7. 반대 부호
negative = tf.negative(node1)

# 8. A > B의 True/False
greater = tf.greater(node1, node2)

# 9. A >= B의 True/False
greater_equal = tf.greater_equal(node1, node2)

# 10. A < B의 True/False
less = tf.less(node1, node2)

# 11. A <= B의 True/False
less_equal = tf.less_equal(node1, node2)

# 12. 반대의 참 거짓
logical = tf.logical_not(True)

# 13. 절대값
abs = tf.abs(node1)
'''

