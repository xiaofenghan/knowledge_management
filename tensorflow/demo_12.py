# -*- coding: utf-8 -*-
"""
create time: 2018-11-15 22:18

author: fnd_xiaofenghan

content: CART classify and Graphviz
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# 用于显示树的结构
from sklearn.tree import export_graphviz


def iris_cart():
    iris = load_iris()
    # x 花瓣的长和宽
    X = iris['data'][:, 2:]
    # 花的类型
    y = iris['target']
    # 建立简单的cart，默认采用gini计算节点
    tree = DecisionTreeClassifier(max_depth=2)
    # 计算
    tree.fit(X, y)

    # 生成可视化文件 .dot
    export_graphviz(tree, out_file='tree_graph.dot', feature_names=iris['feature_names'][2:], class_names=iris['target_names'])

    # 需要在windows上安装 graphviz
    # 然后将 graphviz 的 bin 路径添加到系统环境变量中 测试 cmd 下输入 dot -version
    # 成功后在 tree_graph.dot 文件目录下运行 dot -Tpdf tree_graph.dot -o xxx.pdf



if __name__ == '__main__':
    iris_cart()
