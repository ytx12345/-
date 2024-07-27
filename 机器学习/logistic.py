import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Logistic():
    def __init__(self, train_x, train_y):
        '''

        :param train_x: 训练集
        :param train_y: 训练集类别
        '''
        self.train_x = np.mat(train_x) #数据
        self.train_y = np.mat(train_y) #类别
        self.n_class = 3 #类别数
        self.n_samples, self.n_features = self.train_x.shape  #样本数 特征数

    #将类别进行独热编码
    def one_hot(self,y):
        '''

        :param y: 类别
        :return:
        '''
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.n_class))
        one_hot[np.arange(n_samples), y.T] = 1

        return one_hot

    #将预测结果转化成概率
    def softmax(self, scores):
        '''

        :param scores: 将预测的结果
        :return:
        '''

        # 计算总和
        sum_exp = np.sum(np.exp(scores), axis=1)
        # 计算各个类别概率
        p = np.exp(scores) / sum_exp

        return p

    #训练模型
    def fit(self, iters = 1000, alpha=0.1):
        '''

        :param iters: 迭代步数
        :param alpha: 步长
        :return:
        '''

        # 随机初始化权重矩阵
        self.weights = np.random.rand(self.n_class, self.n_features)

        # 计算 one-hot 矩阵(独热编码)
        y_one_hot = self.one_hot(self.train_y)


        # 定义所有损失结果
        self.all_loss = []
        for i in range(iters):  # 迭代次数
            scores = self.train_x * self.weights.T
            # 计算softmax的值(计算各个类别的概率)
            probs = self.softmax(scores)
            # 代价函数    np.multiply对应位置相乘，求和
            loss = -(1.0 / self.n_samples) * np.sum(np.multiply(y_one_hot, np.log(probs)))
            self.all_loss.append(loss)
            # 求解梯度
            grad = -(1.0 / self.n_samples) * ((y_one_hot - probs)).T * self.train_x
            # 更新权重矩阵
            self.weights = self.weights - alpha * grad

    #预测数据
    def predict(self, test_x):
        '''

        :param test_x: 测试集
        :return:
        '''
        scores = test_x * self.weights.T
        probs = self.softmax(scores)

        return np.argmax(probs, axis=1)  # 返回测试集预测的类别

    #模型评价
    def test(self, test_x, test_y):
        '''

        :param test_x: 测试集
        :param test_y: 从测试集类别
        :return:
        '''

        y_predict = self.predict(test_x)
        accuray = np.sum(y_predict == test_y) / len(test_y)

        print("测试集正确率：%.4f" % accuray)

        # 绘制损失函数
        plt.figure()
        plt.title("loss - train iter")
        plt.xlabel("train iter")
        plt.ylabel("loss")
        plt.plot(np.arange(1000), self.all_loss)
        plt.show()


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #特征名
    target = iris_data['target']  #样本类型
    data = iris_data['data']  #数据集
    target = target.reshape(150, 1)
    train_x, test_x,  train_y, test_y = train_test_split(data, target, test_size = 0.2, random_state = 7)

    logistic = Logistic(train_x, train_y)
    logistic.fit()
    logistic.test(test_x, test_y)