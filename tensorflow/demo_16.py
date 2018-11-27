# -*- coding: utf-8 -*-
"""
create time: 2018-11-26 22:54

author: fnd_xiaofenghan

content:
"""

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score


def compare_models():
    X, y = make_classification(n_samples=10000)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 对lr部分也给出训练集
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_train, y_train, test_size=0.5)

    # 建立模型
    n_estimaters = 100
    gbc = GradientBoostingClassifier(n_estimators=n_estimaters)
    encoder = OneHotEncoder()
    lr = LogisticRegression()

    # 训练决策树
    gbc.fit(X_train, y_train)

    # encode 编码规则训练
    # apply返回每个树最终落到那个叶子上 apply返回的是 所在样本落在第几个树的第几个叶上
    # 注意apply返回的维度
    encoder.fit(gbc.apply(X_train)[:, :, 0])

    # 训练Logistic 单独采用一些样本
    lr.fit(encoder.transform(gbc.apply(X_train_lr)[:, :, 0]), y_train_lr)

    # predict
    # 预测概率 .predict_proba 返回的是每个类里面的概率，只需要选择正类的概率即可
    y_test_pred = lr.predict_proba(encoder.transform(gbc.apply(X_test)[:, :, 0]))[:, 1]

    # plot roc
    # roc的分类只能鉴别两类 并且必须预测结果必须能以秩排序
    fpr_gbc_lr, tpr_gbc_lr, _ = roc_curve(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)
    print(auc)

    # make roc graph
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1], 'k-')
    ax.plot(fpr_gbc_lr, tpr_gbc_lr, label='gbc-lr')


    # 若只采用lr回归预测看看效果呢
    lr.fit(X_train_lr, y_train_lr)
    y_test_pred_lr = lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_pred_lr)
    ax.plot(fpr_lr, tpr_lr, label='lr')

    # 若只采用gbc来预测呢
    y_test_pred_gbc = gbc.predict_proba(X_test)[:, 1]
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_test_pred_gbc)
    ax.plot(fpr_gbc, tpr_gbc, label='gbc')


    # 呈现
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    compare_models()