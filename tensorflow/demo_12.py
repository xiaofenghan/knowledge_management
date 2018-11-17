# -*- coding: utf-8 -*-
"""
create time: 2018-11-15 22:18

author: fnd_xiaofenghan

content: CART classify regression and Graphviz

决策树的众多特性之一就是， 它不需要太多的数据预处理， 尤其是不需要进行特征的缩放或者归一化
它很容易理解和解释，易于使用且功能丰富而强大。然而，它也有一些限制
首先，你可能已经注意到了，决策树很喜欢设定正交化的决策边界，
（所有边界都是和某一个轴相垂直的），这使得它对训练数据集的旋转很敏感。
解决这个难题的一种方式是使用 PCA，这样通常能使训练结果变得更好一些。

"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# 用于显示树的结构
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt


def cart_classify():
    iris = load_iris()
    # x 花瓣的长和宽
    X = iris['data'][:, 2:]
    print(X.shape)

    # 花的类型
    y = iris['target']
    # 建立简单的cart分类树，默认采用gini计算节点
    tree = DecisionTreeClassifier(max_depth=2)

    # 计算
    tree.fit(X, y)

    # 预测
    # y_new = tree.predict(X_new)

    # 生成可视化文件 .dot
    export_graphviz(tree, out_file='tree_graph.dot', feature_names=iris['feature_names'][2:], class_names=iris['target_names'])

    # 需要在windows上安装 graphviz
    # 然后将 graphviz 的 bin 路径添加到系统环境变量中 测试 cmd 下输入 dot -version
    # 成功后在 tree_graph.dot 文件目录下运行 dot -Tpdf tree_graph.dot -o xxx.pdf


def cart_regression():
    # 回归树用于给出连续型输出
    noise = np.random.rand(1000)*10
    X = np.linspace(-5, 5, 1000)
    y = X**2 + noise
    tree_reg = DecisionTreeRegressor(max_depth=3)
    tree_reg.fit(X, y)

    # 预测
    # tree_reg.predict(X_new)

if __name__ == '__main__':
    # cart_classify()
    cart_regression()