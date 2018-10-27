# -*- coding: utf-8 -*-
"""
create time: 2018-10-21 21:57

author: fnd_xiaofenghan

content: 一个简单的神经网络 一些画图技巧
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 增加层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
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

    # 画图真实数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    # 使用plt.ion() 可以不阻塞画图，不然会阻塞到show()，这样可以后面进行动态画图
    plt.ion()
    plt.show()

    # 建立网络 由于是N行1列输入，所以采用[None,1]
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    L_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(L_1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)


    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 20 == 0:
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            # 增加拟合曲线
            lines = ax.plot(x_data, prediction_value, 'r-', lw=1)
            # 暂停图形
            plt.pause(0.1)
            # 去除掉第一条线
            ax.lines.remove(lines[0])

    input()
