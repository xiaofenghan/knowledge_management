# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:34

author: fnd_xiaofenghan

content: 使用placeholder
"""
import tensorflow as tf

input1 = tf.placeholder(tf.float32, [2, 2])
input2 = tf.placeholder(tf.float32, [2, 2])
# 点乘
output = tf.multiply(input1, input2)
# 矩阵乘法
# output = tf.matmul(input1, input2)

with tf.Session() as sess:
    # placeholder的run需要给出feed_dict
    print(sess.run(output, feed_dict={input1: [[1, 1], [1, 2]], input2: [[3, 3], [2,2]]}))