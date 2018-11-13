# -*- coding: utf-8 -*-
"""
create time: 2018-11-06 23:10

author: fnd_xiaofenghan

content: svm using gaussian kernel
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets


if __name__ == '__main__':
    # x_data是样本 y_data是类别
    (x_data, y_data) = datasets.make_circles(n_samples=500, factor=0.5, noise=0.1)

    # 将y=0,1 换成 -1,1
    y_data = np.asarray([s if s == 1 else -1 for s in y_data])

    # 分别画出两个样本 实际上都是x_data, x_data是一个n,2的矩阵
    class1_x = [s[0] for i, s in enumerate(x_data) if y_data[i] == 1]
    class1_y = [s[1] for i, s in enumerate(x_data) if y_data[i] == 1]
    class2_x = [s[0] for i, s in enumerate(x_data) if y_data[i] == -1]
    class2_y = [s[1] for i, s in enumerate(x_data) if y_data[i] == -1]

    # 查看一下图形
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(class1_x, class1_y, 'r+', label='class_1')
    ax.plot(class2_x, class2_y, 'b+', label='class_2')
    ax.legend(loc='lower right')
    # plt.show()

    # 建立网络
    x_input = tf.placeholder(tf.float32, shape=[None, 2])
    y_input = tf.placeholder(tf.float32, shape=[None, 1])

    # batch_size
    batch_size = 350

    # 给预测点分配一下占位符，用于后面画图
    x_input_new = tf.placeholder(tf.float32, shape=[None, 2])
    # svm的对偶问题所需要求解的变量
    b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    # 高斯核函数 exp(-gamma*||x_i - x_j||**2)
    gamma = tf.constant(-50.)
    # 将||x_i - x_j|| 展开就是下面
    dist = tf.reduce_sum(tf.square(x_input), 1)
    # 需要将dist变为batchsize行1列
    dist = tf.reshape(dist, [-1, 1])
    # 二范数展开里面就是这样，虽然dist是1列，但是tf可以自动broadcast
    part = dist - 2*tf.matmul(x_input, tf.transpose(x_input)) + tf.transpose(dist)
    gaussian_kernel = tf.exp(gamma*tf.abs(part))

    # 损失函数
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_input_cross = tf.matmul(y_input, tf.transpose(y_input))
    second_term = tf.reduce_sum(tf.multiply(gaussian_kernel, tf.multiply(b_vec_cross, y_input_cross)))
    loss = tf.negative(first_term - second_term)

    # 预测新点分类
    dist_old = tf.reshape(tf.reduce_sum(tf.square(x_input), 1), [-1, 1])
    dist_new = tf.reshape(tf.reduce_sum(tf.square(x_input_new), 1), [-1, 1])

    pred_part = dist_old - 2*tf.matmul(x_input, tf.transpose(x_input_new)) + tf.transpose(dist_new)
    pred_kernel = tf.exp(gamma*tf.abs(pred_part))
    pred_output = tf.matmul(tf.multiply(tf.transpose(y_input), b), pred_kernel)

    # 预测
    prediction = tf.sign(pred_output - tf.reduce_mean(pred_output))

    #
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_input)), tf.float32))

    # 
    opt = tf.train.GradientDescentOptimizer(0.002)
    step = opt.minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    #
    loss_ = []
    acc_ = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_data), size=batch_size)
        rand_x = x_data[rand_index]
        rand_y = y_data[rand_index][:, np.newaxis]
        sess.run(step, feed_dict={x_input: rand_x, y_input: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_input: rand_x, y_input: rand_y})
        temp_acc = sess.run(acc, feed_dict={x_input: rand_x, y_input: rand_y, x_input_new: rand_x})
        loss_.append(temp_loss)
        acc_.append(temp_acc)

        if i % 250 == 0:
            print(i, temp_loss, temp_acc)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.asarray(loss_), 'k-', label='loss')
    ax1.legend(loc='Best')

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(np.asarray(acc_), 'k-', label='acc')
    ax2.legend(loc='Best')
    plt.show()


