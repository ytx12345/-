import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Gaussian():
    def __init__(self):

        self.priors = None  #先验概率
        self.means = None  #均值
        self.vars = None  #方差

    # （1）求训练集各个类别的先验概率
    def prior_predict(self, train_y):
        '''

        :param train_y: 训练集
        :return:
        '''
        prior = []  #各类别的先验概率
        classify = set(train_y) #类别 {0,1,2}
        n_samples = len(train_y)  #样本数

        for label in classify:
            prior.append(np.sum(train_y[:] == label) / n_samples)

        return prior

    #（2）求各个类别的期望和方差
    def mean_cov(self, train_x, train_y):
        '''

        :param train_x: 训练集
        :param train_y: 训练集对应的类别
        :return:
        '''
        mean = [] #期望
        cov_matrix = [] #方差

        for label in set(train_y):
            cov = np.zeros((4,4))
            label_data = train_x[train_y == label]
            mean.append(np.mean(label_data, axis=0)) #求每个类别的每个属性期望
            (cov[0, 0], cov[1, 1], cov[2, 2], cov[3, 3]) = np.var(label_data, axis=0)
            cov_matrix.append(cov) #求每个类别每个属性方差

        return mean, cov_matrix

    #（3）计算类条件概率
    def class_probability(self, x, mean, cov):
        '''

        :param x: 待预测数据
        :param mean: 均值
        :param cov: 方差
        :return:
        '''
        x1 = np.mat(x).T  # 待测试样本
        mean1 = np.mat(mean).T  # 期望
        cov1 = np.mat(cov)  # 协方差

        m = -((x1 - mean1).T * cov1.I * (x1 - mean1)) / 2  # 类条件概率函数的指数部
        n = 2 * 3.14 * (np.linalg.det(cov1) ** 1 / 2)  # 类条件概率的的常数部分
        p = (1 / n) * np.math.exp(np.linalg.det(m))  # 类条件概率的预测值

        return p
    #（4）预测类别
    def predict(self, x):
        '''

        :param x: 待预测数据
        :return:
        '''
        labels = []
        for mean, cov, prior in zip(self.means, self.vars, self.priors):
            pre = self.class_probability(x, mean, cov)
            pre = pre * prior
            labels.append(pre)

        return np.argmax(labels)

    #模型评价
    def fit(self, train_x, train_y):
        '''

        :param train_x: 训练集
        :param train_y: 训练集对应的类别
        :return:
        '''

        self.priors = self.prior_predict(train_y) #先验概率
        self.means, self.vars = self.mean_cov(train_x, train_y) #期望、方差

    #模型的评价
    def test(self, test_x, test_y):
        '''

        :param test_x: 测试集
        :param test_y: 测试集对应的类别
        :return:
        '''

        predicts = []  #预测类别

        for x in test_x:
            predicts.append(self.predict(x))

        accuray = np.sum(predicts == test_y) / len(test_y)  #正确率

        print("测试集正确率：%.4f" % accuray)


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #特征名
    target = iris_data['target']  #样本类型
    data = iris_data['data']  #数据集

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.1, random_state = 0)

    gaussian = Gaussian()
    gaussian.fit(train_x, train_y)
    gaussian.test(test_x, test_y)