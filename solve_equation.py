#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
"""Basic usage of tensorflow
solve eqation Ax = y
A = [1, 2]
    [3, 4]
y = [10]
    [22]
so x = ?
"""

if __name__ == '__main__':
    A = tf.constant( [[1., 2.], [3., 4.]])
    y = tf.constant([[10.], [22.]])
    x = tf.Variable(tf.zeros([2, 1])) # x is unkown, so set it a Variable
    y_ = tf.matmul(A, x)
    """ y is the target value
        y_ is current value
        so, less (y_ - y) indicates more accurate x
    """

    deviation = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(deviation)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        sess.run(train_step)
    print(sess.run(x))
