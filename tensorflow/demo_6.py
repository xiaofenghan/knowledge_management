# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:57

author: fnd_xiaofenghan

content: 一个简单的神经网络 tensorboard的演示
"""
import tensorflow as tf
import numpy as np


# 增加层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name='Weight')
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')
        Wx_plus_b = tf.matmul(inputs, W) + b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

    return outputs


if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise


    # 建立网络 由于是N行1列输入，所以采用[None,1]
    # 建立节点，用于tensorboard的展示，节点使用name_scope
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    L_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(L_1, 10, 1, activation_function=None)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

    with tf.name_scope('Train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    # 一定要在session定义完毕之后将图层写入到log中
    writer = tf.summary.FileWriter('logs/', graph=sess.graph)

    sess.run(init)


    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 20 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


    # 运行完毕后，到目录'logs' 打开cmd
    # 运行 tensorboard --logdir logs
    # 复制cmd出现的地址到chrome中