# -*- coding：utf-8 -*-
import tensorflow as tf
def sigmoid(z):
    x = tf.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})
    return result