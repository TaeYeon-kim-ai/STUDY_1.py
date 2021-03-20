import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3) #tf.Tensor(7.0, shape=(), dtype=float32)

sess = tf.Session()
print('sess.run(node1, noder2)', sess.run([node1, node2]))sess.run(node3) : ', sess.run(node3)) #sess.run(node3) :  7.0

#텐서는 다차원을 끌어오는역할을 하고
#그러기 위해서 다차원 배열을 만든다datetime A combination of a date and a time. Attributes: ()