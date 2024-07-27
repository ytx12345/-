import numpy as np
from sklearn.datasets import load_iris
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #标准化预处理
from sklearn.model_selection import train_test_split

class K_means():
    def __init__(self, k):
        '''

        :param k:  将数据集划分为k簇
        '''
        self.K = k

    # 随机选取k个中心点
    def find_center(self):
        center = []
        for i in range(self.K):
            index = np.random.randint(self.x_std.shape[0])
            center.append(self.x_std[index])
        return center

    def fit(self, train_x,iters = 100):
        '''

        :param train_x:  数据集
        :param iters: 迭代次数
        :return:
        '''
        self.train_x = train_x
        #对数据进行标准化预处理
        scaler = StandardScaler().fit(self.train_x)
        self.x_std = scaler.transform(self.train_x)
        n_samples, n_features = self.x_std.shape  #样本数 特征数

        self.center = self.find_center()  #中心点
        self.train_y = np.zeros(n_samples, dtype=np.int32)  #样本类别


        for iter in range(iters):  #迭代次数
            for i, x in enumerate(self.x_std): #遍历每个样本点
                distance = [] #存放样本点x距离各个中心点的欧式距离
                for k in self.center:  #遍历每个中心
                    # 计算样本点与中心点的欧式距离(numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2))))
                    distance.append(np.linalg.norm(x - k))
                label = np.array(distance).argmin()  #样本类别
                self.train_y[i] = label
                #更新中心点
                self.center[label] = np.mean(self.x_std[self.train_y == label,:],axis=0)

    def data_plot(self, data, target):
        '''

        :param data: 原始数据
        :param target:原始类别
        :return:
        '''

        # 设置画布
        plt.figure(figsize=(10, 5),dpi=100)

        #原始图
        plt.subplot(1, 2, 1)
        plt.title("original")
        plt.xlabel("sepal length (cm)")
        plt.ylabel("sepal width (cm)")
        plt.scatter(data[target == 0, 2], data[target == 0, 3], color='r')
        plt.scatter(data[target == 1, 2], data[target == 1, 3], color='b')
        plt.scatter(data[target == 2, 2], data[target == 2, 3], color='g')

        #k-means聚类后
        plt.subplot(1, 2, 2)
        plt.title("k_means")
        plt.xlabel("sepal length (cm)")
        plt.ylabel("sepal width (cm)")
        plt.scatter(self.train_x[self.train_y == 0, 2], self.train_x[self.train_y == 0, 3], color='r')
        plt.scatter(self.train_x[self.train_y == 1, 2], self.train_x[self.train_y == 1, 3], color='b')
        plt.scatter(self.train_x[self.train_y == 2, 2], self.train_x[self.train_y == 2, 3], color='g')
        plt.show()


#打乱数据集
def random(data, target):
    index = [i for i in range(len(data))]
    shuffle(index)  #打乱索引
    data  = data[index] #打乱数据集
    target = target[index]
    return data,target


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #特征名
    target = iris_data['target']  #样本类型
    data = iris_data['data']  #数据集
    data,target = random(data, target)

    k_means = K_means(3)
    k_means.fit(data)
    k_means.data_plot(data,target)