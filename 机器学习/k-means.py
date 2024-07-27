import numpy as np
from sklearn.datasets import load_iris
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #��׼��Ԥ����
from sklearn.model_selection import train_test_split

class K_means():
    def __init__(self, k):
        '''

        :param k:  �����ݼ�����Ϊk��
        '''
        self.K = k

    # ���ѡȡk�����ĵ�
    def find_center(self):
        center = []
        for i in range(self.K):
            index = np.random.randint(self.x_std.shape[0])
            center.append(self.x_std[index])
        return center

    def fit(self, train_x,iters = 100):
        '''

        :param train_x:  ���ݼ�
        :param iters: ��������
        :return:
        '''
        self.train_x = train_x
        #�����ݽ��б�׼��Ԥ����
        scaler = StandardScaler().fit(self.train_x)
        self.x_std = scaler.transform(self.train_x)
        n_samples, n_features = self.x_std.shape  #������ ������

        self.center = self.find_center()  #���ĵ�
        self.train_y = np.zeros(n_samples, dtype=np.int32)  #�������


        for iter in range(iters):  #��������
            for i, x in enumerate(self.x_std): #����ÿ��������
                distance = [] #���������x����������ĵ��ŷʽ����
                for k in self.center:  #����ÿ������
                    # ���������������ĵ��ŷʽ����(numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2))))
                    distance.append(np.linalg.norm(x - k))
                label = np.array(distance).argmin()  #�������
                self.train_y[i] = label
                #�������ĵ�
                self.center[label] = np.mean(self.x_std[self.train_y == label,:],axis=0)

    def data_plot(self, data, target):
        '''

        :param data: ԭʼ����
        :param target:ԭʼ���
        :return:
        '''

        # ���û���
        plt.figure(figsize=(10, 5),dpi=100)

        #ԭʼͼ
        plt.subplot(1, 2, 1)
        plt.title("original")
        plt.xlabel("sepal length (cm)")
        plt.ylabel("sepal width (cm)")
        plt.scatter(data[target == 0, 2], data[target == 0, 3], color='r')
        plt.scatter(data[target == 1, 2], data[target == 1, 3], color='b')
        plt.scatter(data[target == 2, 2], data[target == 2, 3], color='g')

        #k-means�����
        plt.subplot(1, 2, 2)
        plt.title("k_means")
        plt.xlabel("sepal length (cm)")
        plt.ylabel("sepal width (cm)")
        plt.scatter(self.train_x[self.train_y == 0, 2], self.train_x[self.train_y == 0, 3], color='r')
        plt.scatter(self.train_x[self.train_y == 1, 2], self.train_x[self.train_y == 1, 3], color='b')
        plt.scatter(self.train_x[self.train_y == 2, 2], self.train_x[self.train_y == 2, 3], color='g')
        plt.show()


#�������ݼ�
def random(data, target):
    index = [i for i in range(len(data))]
    shuffle(index)  #��������
    data  = data[index] #�������ݼ�
    target = target[index]
    return data,target


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #������
    target = iris_data['target']  #��������
    data = iris_data['data']  #���ݼ�
    data,target = random(data, target)

    k_means = K_means(3)
    k_means.fit(data)
    k_means.data_plot(data,target)