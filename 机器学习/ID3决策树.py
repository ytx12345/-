'''
֪ʶ�㣺https://www.cnblogs.com/sxron/p/5471078.html
��1����������ѧϰ���̣�
    ����ѡ������ѡ����ָ��ѵ���������ڶ��������ѡ��һ��������Ϊ��ǰ�ڵ�ķ��ѱ�׼�����ѡ���������źܶ಻ͬ����������׼��׼���Ӷ���������ͬ�ľ������㷨��
    ���������ɣ� ����ѡ�������������׼���������µݹ�������ӽڵ㣬ֱ�����ݼ����ɷ���ֹͣ������ֹͣ������ ���ṹ��˵���ݹ�ṹ�����������ķ�ʽ��
    ��֦�����������׹���ϣ�һ������Ҫ��֦����С���ṹ��ģ���������ϡ���֦������Ԥ��֦�ͺ��֦���֡�
��2����ɢ������ID3(��Ϣ����)     ����������CART(��Ϣ����)��C4.5��GINIָ����
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class DecisionTree():

    def __init__(self):
        pass

    #��1����һ�����ݼ�����ĳһ�����Ե�ָ��ֵ���л���
    def pre_splite(self, index, value, dataset):
        """
        @index: ָ������
        @value: ָ�����ֵ���ֵ
        @dataset: ���ݼ�
        """
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    #��2��
    #������Ϣ��
    def entropy(self, group, classes):
        """
        @groups: ���ֺ�����ݼ�
        @classes: ��𼯺�
        """
        m = len(group)  #��ǰ����Ĵ�С
        p = 0
        for val in classes:
            sub_data_p = [row[-1] for row in group].count(val) / m #������������ռ�ı���
            if sub_data_p==0:
                continue
            p += (sub_data_p)* np.log2(sub_data_p)  #������Ϣ����

        return -p

    #������Ϣ����
    def ent_index(self, groups, classes):
        """
        @groups: ���ֺ�����ݼ�
        @classes: ��𼯺�
        """
        # ����ȫ���Ĵ�С
        n_instances = float(sum([len(group) for group in groups]))

        #δ����֮ǰ�����Ϣ��
        Ent = self.entropy(groups[0]+groups[1], classes)

        #���ֺ�ÿһ���Ӽ��ļ�Ȩ��Ϣ��
        ent = 0.0
        for group in groups:
            size = float(len(group))  # ��ǰ�Ӽ��Ĵ�С
            # ע�⣺��ĸ����Ϊ0
            if size == 0:
                continue
            weight = (size / n_instances)
            score = self.entropy(group, classes)
            ent += weight * score  # ���㱻���ֹ������ݼ���giniָ��

        return Ent-ent

    #��3���ҵ���õķָ��
    def get_split(self, dataset):
        """
        @dataset: ���ݼ�
        """
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('-inf'), None
        for index in range(len(dataset[0]) - 1): #ĳ������
            for row in dataset: #ÿ������
                groups = self.pre_splite(index, row[index], dataset) #�����ݼ�����ĳһ�����Ե�ָ��ֵ���л���
                ent = self.ent_index(groups, class_values) #���㱻���ֵ��Ӽ��Ļ���ϵ��
                if ent > b_score: #���giniָ���ȵ�ǰֵС�����������ֵ����ݼ�
                    b_index, b_value, b_score, b_groups = index, row[index], ent, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}  # ��dict��ʽ���ؽ��

    #��4��Ҷ�ӽڵ����ֵ
    def to_terminal(self, group):
        """
        @groups: ��
        """
        outcomes = [row[-1] for row in group] #Ҷ�ӽڵ�����ֵ����
        return max(set(outcomes), key=outcomes.count) #�����ǰҶ�ӽڵ����������ΪԤ�����

    #��5�����ķֲ�
    def split(self, node, max_depth, min_size, depth):
        """
        @node: ��ǰ���ڵĽڵ�
        @max_depth: ������
        @min_size: ֹͣ��ֵ���Ŀ����ֹ������
        @depth: ���
        """
        left, right = node['groups']  #��ǰ�ڵ�������ӽڵ�

        del (node['groups'])  # ɾ��
        if not left or not right: # �Ѿ�����Ҷ��
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:  #�Ѿ��ﵽ������
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:  #�������Ѿ��ﵽҶ�ӽڵ����ٵ�������
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left) #�����ݹ鹹��������
            self.split(node['left'], max_depth, min_size, depth + 1)
        if len(right) <= min_size:  #�������Ѿ��ﵽҶ�ӽڵ����ٵ�������
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right) #�����ݹ鹹��������
            self.split(node['right'], max_depth, min_size, depth + 1)

    #��6�����
    def build_tree(self, train, max_depth, min_size):
        """
        @train: ѵ����
        @max_depth: ������
        @min_size: ֹͣ��ֵ���Ŀ����ֹ������
        """
        root = self.get_split(train) #���ָ��ڵ�
        self.split(root, max_depth, min_size, 1)  # �ݹ���
        return root

    #(7)���
    def fit(self, train, max_depth=8, min_size=2):
        self.tree = self.build_tree(train, max_depth, min_size)


    #Ԥ�ⵥ������
    def tree_prediction(self, node, row):
        """
        ʹ�þ�����������Ԥ��
        @node: һ���µ����ݵ�
        @row: ��ǰ���ڲ�
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                # �����������δ����Ҷ��������ݹ�
                return self.tree_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.tree_prediction(node['right'], row)
            else:
                return node['right']

    #Ԥ����Լ�
    def predict(self,test):
        """
        @text: ���Լ�
        """
        # Ԥ����Լ�
        predictions = []
        for row in test:
            prediction = self.tree_prediction(self.tree, row)
            predictions.append(prediction)

        test_y = test[:, -1]  #��ȷ���������
        print("�β�����Լ�����ȷ�ʣ�%f" % (np.sum(predictions == test_y) / len(test_y)))