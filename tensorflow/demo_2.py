# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:19

author: fnd_xiaofenghan

content: 使用session
"""
import tensorflow as tf

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [2]])
# 矩阵相乘
product = tf.matmul(m1, m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result = sess.run(product)
print(result)