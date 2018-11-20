# -*- coding: utf-8 -*-
"""
create time: 2018-11-17 15:11

author: fnd_xiaofenghan

content: Random Forest
"""
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
from sklearn.datasets import make_moons

from sklearn.ensemble import BaggingClassifier

# 随机森林是bagging的一个完整例子
from sklearn.ensemble import RandomForestClassifier
# 计算随机森林，不仅仅在属性上进行随机划分，属性的阈值也是随机划分的一个集成学习
from sklearn.ensemble import ExtraTreesClassifier

# 基学习器
from sklearn.tree import DecisionTreeClassifier




def bagging_classify():
    """
    sklearn 为 bagging（有放回抽样） 和 pasting（无放回抽样） 提供了接口
    :return:
    """
    # y 是label
    X, y = make_moons(500, noise=0.1)
    plt.plot(X[:, 0], X[:, 1], 'r+')
    plt.show()

    # 建立一个集成学习 bagging
    bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, max_samples=20, bootstrap=True, n_jobs=-1)
    X_train = X[:400, :]
    X_test = X[400:, :]
    y_train = y[:400]
    y_test = y[400:]

    # 学习
    bag_clf.fit(X_train, y_train)


    # 预测
    y_pred = bag_clf.predict(X_test)
    print('MSE  ', np.sum((y_pred-y_test)**2)/len(y_pred))



def random_forest_classify():
    """
    随机森林有直接的表示方法, 速度快多了
    :return:
    """
    # y 是label
    X, y = make_moons(500, noise=0.1)
    plt.plot(X[:, 0], X[:, 1], 'r+')
    plt.show()

    # 建立一个rf
    rf_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=10, n_jobs=-1)
    # rf_clf = ExtraTreesClassifier(n_estimators=10, max_leaf_nodes=10, n_jobs=-1)
    # 上面也可以采用bagging的方式写出
    # rf_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_leaf_nodes=10, splitter='random'),
    #                            n_estimators=10, max_samples=20, bootstrap=True, n_jobs=-1)

    X_train = X[:400, :]
    X_test = X[400:, :]
    y_train = y[:400]
    y_test = y[400:]

    # 预测
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print('MSE  ', np.sum((y_pred-y_test)**2)/len(y_pred))

    # 计算属性的平均深度，越靠近根部越重要
    print(rf_clf.feature_importances_)




if __name__ == '__main__':
    # bagging_classify()
    random_forest_classify()