"""
一个回归例子
"""

import tensorflow as tf
import numpy as np

# 输入数据
x_data = np.random.normal(1, 0.1, 100)[:, np.newaxis]
y_data = np.repeat(10., 100)[:, np.newaxis]

# 占位符
x_input = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

# 待估参数
A = tf.Variable(tf.random_normal(shape=[1]))
c = tf.Variable(tf.random_normal(shape=[1]))

# 定义误差
output = tf.multiply(A, tf.square(x_input)) + c
loss = tf.reduce_mean(tf.square(output - y_target))

# 优化器部分
optimal = tf.train.GradientDescentOptimizer(0.2)
train_step = optimal.minimize(loss)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 迭代求解
for s in range(1000):
    i = np.random.choice(100)
    x_sample = x_data[: i]
    y_sample = y_data[: i]
    sess.run(train_step, feed_dict={x_input: x_sample, y_target: y_sample})
    if s % 10 == 0:
        print(s, sess.run(loss, feed_dict={x_input: x_data, y_target: y_data}))
