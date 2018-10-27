# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:25

author: fnd_xiaofenghan

content: 使用Variable
"""
import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 一定要先运行 init 这一步
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(i, sess.run(state))