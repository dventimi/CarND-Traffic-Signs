import tensorflow as tf

def add():
    x = tf.Variable(tf.ones([2,2]))
    y = tf.Variable(tf.ones([2,2]))
    val = tf.matmul(x, y)
    return x

sess = tf.Session()
x = add()
sess.run(tf.initialize_all_variables())
print(sess.run(x))
