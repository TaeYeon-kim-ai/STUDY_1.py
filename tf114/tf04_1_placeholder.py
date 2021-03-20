import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

#placeholder --- input느낌
#input지정
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#덧셈정의
adder_node = a + b 
print(sess.run(adder_node, feed_dict={a:3, b:4.5})) #key value형식 딕셔너리 형식
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

#곱셈정의
add_and_triple = adder_node * 3 
print(sess.run(add_ and_triple, feed_dict={a:4, b:2})) #18 // 4와 2더한거에 3곱하기




