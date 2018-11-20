# -*- coding: utf-8 -*-
"""
create time: 2018-11-20 21:59

author: fnd_xiaofenghan

content: GBDT and GBRT
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor
# 计算mse
from sklearn.metrics import mean_squared_error
# 将样本切分成训练和测试
from sklearn.model_selection import train_test_split


from sklearn.ensemble import GradientBoostingRegressor


def GBDT_classify():
    """
    GBDT 的思路 类似于 第t个分类器是 将第t-1个分类器的预测偏差作为y，输入仍为X，串行计算
    :return:
    """
    noise = np.random.uniform(-1, 1, 1000)
    x = np.linspace(-10, 10, 1000) ** 2
    y = x + 10 * noise
    x_train = x[:800][:, np.newaxis]
    y_train = y[:800][:, np.newaxis]
    x_test = x[800:][:, np.newaxis]
    y_test = y[800:][:, np.newaxis]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(y, 'r+')
    # plt.show()

    gbdt = GradientBoostingRegressor(n_estimators=3, learning_rate=1, max_depth=3)
    gbdt.fit(x_train, y_train)
    y_test_pred = gbdt.predict(x_test)[:, np.newaxis]

    print('mse ', mean_squared_error(y_test, y_test_pred))

    # 多少个树是最合适的？使用staged_
    # GradientBoostingRegressor中的learning_rate表示每棵树的学习权重，一般越小同时树越多，学习越精确，因此需要找到一个平衡
    gbrt_find = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    gbrt_find.fit(x_train, y_train)

    # staged_predict 返回一系列 n_estimators 的gbrt模型，是一个迭代器
    # 通过计算error，然后查看图形中 对应于 learning_rate 下的最合适的 n_estimators
    all_pred = gbrt_find.staged_predict(x_test)
    error = []
    for s in all_pred:
        error.append(mean_squared_error(s, y_test))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(error, 'r+')
    plt.show()


def gbdt_step_by_step():
    """
    模拟gbdt的思路，分步骤单独写
    :return:
    """
    noise = np.random.uniform(-1, 1, 1000)
    x = np.linspace(-10, 10, 1000) ** 2
    y = x + 10 * noise

    x_train = x[:800][:, np.newaxis]
    y_train = y[:800][:, np.newaxis]
    x_test = x[800:][:, np.newaxis]
    y_test = y[800:][:, np.newaxis]

    tree_1 = DecisionTreeRegressor(max_depth=3)
    tree_1.fit(x_train, y_train)
    y_2_train = y_train - tree_1.predict(x_train)[:, np.newaxis]


    tree_2 = DecisionTreeRegressor(max_depth=3)
    tree_2.fit(x_train, y_2_train)
    y_3_train = y_2_train - tree_2.predict(x_train)[:, np.newaxis]

    tree_3 = DecisionTreeRegressor(max_depth=3)
    tree_3.fit(x_train, y_3_train)

    # 合并各个分类器结果

    y_test_pred = np.zeros(y_test.shape)
    for s in [tree_1, tree_2, tree_3]:
        y_test_pred += s.predict(x_test)[:, np.newaxis]

    print('mse ', mean_squared_error(y_test, y_test_pred))


if __name__ == '__main__':
    GBDT_classify()
    # gbdt_step_by_step()
