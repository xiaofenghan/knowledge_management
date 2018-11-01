# -*- coding: utf-8 -*-
"""
create time: 2018-10-31 22:19

author: fnd_xiaofenghan

content: lasso/demming regression
"""
import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # data_set in sklearn
    iris = datasets.load_iris()
    x_data = np.asarray([x[3] for x in iris['data']])[:, np.newaxis]
    y_data = np.asarray([y[0] for y in iris['data']])[:, np.newaxis]

    # construct tensor graph
    x_input = tf.placeholder(tf.float32, shape=[None, 1])
    y_input = tf.placeholder(tf.float32, shape=[None, 1])

    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # regression
    y_predict = tf.multiply(x_input, A) + b

    # loss func
    # lasso regression
    # learning_rate = 0.001
    # penalty_coeff = tf.constant(0.5)
    # loss = tf.square(y_predict - y_input) + tf.multiply(penalty_coeff, tf.abs(A))

    # demming regression
    learning_rate = 0.05
    numerator = tf.abs(y_input - (tf.multiply(A, x_input)+b))
    denominator = tf.sqrt(tf.square(A) + 1)
    loss = tf.reduce_mean(tf.divide(numerator, denominator))

    # optimize
    optimal = tf.train.GradientDescentOptimizer(learning_rate)
    opt_step = optimal.minimize(loss)


    # init
    init = tf.global_variables_initializer()

    # run
    sess = tf.Session()
    sess.run(init)

    # solve
    batch_size = 50
    for i in range(10000):
        index = np.random.choice(len(x_data), batch_size)
        x_data_sample = x_data[index]
        y_data_sample = y_data[index]
        sess.run(opt_step, feed_dict={x_input: x_data_sample, y_input: y_data_sample})
        if i % 50 == 0:
            print(i, sess.run(loss, feed_dict={x_input: x_data, y_input: y_data}))

    # plot true scatter
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data, )

    # plot fit line
    slope = sess.run(A)
    intercept = sess.run(b)
    line = ax.plot(x_data, slope * x_data + intercept, 'r--')
    plt.show()
