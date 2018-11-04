# -*- coding: utf-8 -*-
"""
create time: 2018-11-04 13:44

author: fnd_xiaofenghan

content:
1) soft_margin svm demo using soft-margin loss function  mean(sum(max(0, 1-y*(Ax-b)))) + a*A**2
2) linear svm demo using linear loss function mean(sum(max(0, abs(y-(Ax-b))-eps)))
"""
import numpy as np
import tensorflow as tf
import requests
from sklearn import datasets
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 加载数据
    iris = datasets.load_iris()
    x_data = np.asarray([[x[0], x[3]] for x in iris['data']])
    y_data = np.asarray([1 if y == 0 else -1 for y in iris['target']])

    # 使用80%作为训练 20%作为测试
    # 虽然抽80%，但是80%中可能是存在重复样本的，这取决于np.random.choice函数
    # 所以20%也不代表是剩下的20%
    train_index = np.random.choice(len(x_data), round(0.8*len(x_data)))
    test_index = np.asarray(list(set(range(len(x_data))) - set(train_index)))
    x_data_train = x_data[train_index]
    y_data_train = y_data[train_index][:, np.newaxis]
    x_data_test = x_data[test_index]
    y_data_test = y_data[test_index][:, np.newaxis]


    # 建立网络
    # 因为x中有两个属性，所以列是2，第一个维度表示batchsize，每次进行训练的大小
    x_input = tf.placeholder(tf.float32, shape=[None, 2])
    y_input = tf.placeholder(tf.float32, shape=[None, 1])

    A = tf.Variable(initial_value=tf.random_normal(shape=[2, 1]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))

    y_output = tf.matmul(x_input, A) - b

    # 模型估计准确度计算
    y_prediction = tf.sign(y_output)
    accuracy = tf.reduce_mean(tf.cast((tf.equal(y_prediction, y_input)), tf.float32))

    # soft-margin loss func
    # alpha = tf.constant(0.1)
    # loss = tf.reduce_mean(tf.maximum(0., 1 - tf.multiply(y_input, y_output))) + tf.multiply(alpha, tf.norm(A, ord=2))

    # linear svm loss func
    eps = tf.constant(0.5)
    loss = tf.reduce_mean(tf.maximum(0., tf.abs(y_input - y_output))-eps)

    # optimize
    opt = tf.train.GradientDescentOptimizer(0.01)
    run_opt = opt.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    sess = tf.Session()
    sess.run(init)

    # compute and evaluation
    batch_size = 100
    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    for i in range(200):
        random_index = np.random.choice(len(x_data_train), batch_size)
        x_rand = x_data_train[random_index]
        y_rand = y_data_train[random_index]
        # 计算优化函数
        sess.run(run_opt, feed_dict={x_input: x_rand, y_input: y_rand})
        # 损失函数值
        loss_temp = sess.run(loss, feed_dict={x_input: x_rand, y_input: y_rand})
        # 准确度
        train_acc = sess.run(accuracy, feed_dict={x_input: x_data_train, y_input: y_data_train})
        test_acc = sess.run(accuracy, feed_dict={x_input: x_data_test, y_input: y_data_test})

        # 保存序列
        loss_vec.append(loss_temp)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        print(i)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(loss_vec, label='loss')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(train_accuracy, label='train_acc')
    ax2.plot(test_accuracy, 'r--', label='test_acc')
    ax2.legend(loc='lower right')
    plt.show()
