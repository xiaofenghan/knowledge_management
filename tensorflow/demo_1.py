# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:06

author: fnd_xiaofenghan

content: 最简单的demo
"""

import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data*0.1+0.3

# tensorflow structure
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W*x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 只要有定义了tf.variable就需要初始化函数
init = tf.global_variables_initializer()

# active tensorflow
sess = tf.Session()
# very important, sess.run 相当于指针，激活括号内的变量或算子
sess.run(init)

for i in range(2000):
    sess.run(train)
    if i % 20 == 0:
        print(i, sess.run(W), sess.run(b))