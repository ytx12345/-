'''
知识点：https://www.cnblogs.com/sxron/p/5471078.html
（1）决策树的学习过程：
    特征选择：特征选择是指从训练数据中众多的特征中选择一个特征作为当前节点的分裂标准，如何选择特征有着很多不同量化评估标准标准，从而衍生出不同的决策树算法。
    决策树生成： 根据选择的特征评估标准，从上至下递归地生成子节点，直到数据集不可分则停止决策树停止生长。 树结构来说，递归结构是最容易理解的方式。
    剪枝：决策树容易过拟合，一般来需要剪枝，缩小树结构规模、缓解过拟合。剪枝技术有预剪枝和后剪枝两种。
（2）离散特征：ID3(信息增益)     连续特征：CART(信息增益比)、C4.5（GINI指数）
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class DecisionTree():

    def __init__(self):
        pass

    #（1）将一个数据集根据某一个属性的指定值进行划分
    def pre_splite(self, index, value, dataset):
        """
        @index: 指定属性
        @value: 指定划分的阈值
        @dataset: 数据集
        """
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right


    #（2）计算被划分的子集的系数
    #基尼指数
    def gini_index(self, groups, classes):
        """
        @groups: 划分后的数据集
        @classes: 类别集合
        """
        # 计算全集的大小
        n_instances = float(sum([len(group) for group in groups]))

        # 求每一个子集的加权基尼系数
        gini = 0.0
        for group in groups:
            size = float(len(group))  #当前子集的大小
            # 注意：分母不能为0
            if size == 0:
                continue
            score = 0.0
            for val in classes:
                p = [row[-1] for row in group].count(val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances) #计算被划分过后数据集的gini指数

        return gini

    #（3）找到最好的分割点
    def get_split(self, dataset):
        """
        @dataset: 数据集
        """
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
        for index in range(len(dataset[0]) - 1): #某个属性
            for row in dataset: #每个样本
                groups = self.pre_splite(index, row[index], dataset) #将数据集根据某一个属性的指定值进行划分
                gini = self.gini_index(groups, class_values) #计算被划分的子集的基尼系数
                if gini < b_score: #如果gini指数比当前值小，更新所划分的数据集
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}  # 以dict形式返回结果

    #（4）叶子节点输出值
    def to_terminal(self, group):
        """
        @groups: 组
        """
        outcomes = [row[-1] for row in group] #叶子节点的输出值集合
        return max(set(outcomes), key=outcomes.count) #输出当前叶子节点最多的类别作为预测类别

    #（5）树的分叉
    def split(self, node, max_depth, min_size, depth):
        """
        @node: 当前所在的节点
        @max_depth: 最大深度
        @min_size: 停止拆分的数目（终止条件）
        @depth: 深度
        """
        left, right = node['groups']  #当前节点的左右子节点

        del (node['groups'])  # 删掉
        if not left or not right: # 已经到达叶子
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:  #已经达到最大深度
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:  #左子树已经达到叶子节点最少的样本数
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left) #继续递归构建左子树
            self.split(node['left'], max_depth, min_size, depth + 1)
        if len(right) <= min_size:  #右子树已经达到叶子节点最少的样本数
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right) #继续递归构建右子树
            self.split(node['right'], max_depth, min_size, depth + 1)

    #（6）搭建树
    def build_tree(self, train, max_depth, min_size):
        """
        @train: 训练集
        @max_depth: 最大深度
        @min_size: 停止拆分的数目（终止条件）
        """

        root = self.get_split(train) #划分根节点
        self.split(root, max_depth, min_size, 1)  # 递归搭建树
        return root

    #(7)搭建树
    def fit(self, train, max_depth=10, min_size=2):
        self.tree = self.build_tree(train, max_depth, min_size)


    #预测单个样本
    def tree_prediction(self, node, row):
        """
        使用决策树来进行预测
        @node: 一个新的数据点
        @row: 当前所在层
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                # 如果左子树尚未到达叶子则继续递归
                return self.tree_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.tree_prediction(node['right'], row)
            else:
                return node['right']

    #预测测试集
    def predict(self,test):
        """
        @text: 测试集
        """
        # 预测测试集
        predictions = []
        for row in test:
            prediction = self.tree_prediction(self.tree, row)
            predictions.append(prediction)

        test_y = test[:, -1]  #正确的样本类别
        print("鸢尾花测试集的正确率：%f" % (np.sum(predictions == test_y) / len(test_y)))