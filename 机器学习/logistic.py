import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Logistic():
    def __init__(self, train_x, train_y):
        '''

        :param train_x: ѵ����
        :param train_y: ѵ�������
        '''
        self.train_x = np.mat(train_x) #����
        self.train_y = np.mat(train_y) #���
        self.n_class = 3 #�����
        self.n_samples, self.n_features = self.train_x.shape  #������ ������

    #�������ж��ȱ���
    def one_hot(self,y):
        '''

        :param y: ���
        :return:
        '''
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.n_class))
        one_hot[np.arange(n_samples), y.T] = 1

        return one_hot

    #��Ԥ����ת���ɸ���
    def softmax(self, scores):
        '''

        :param scores: ��Ԥ��Ľ��
        :return:
        '''

        # �����ܺ�
        sum_exp = np.sum(np.exp(scores), axis=1)
        # �������������
        p = np.exp(scores) / sum_exp

        return p

    #ѵ��ģ��
    def fit(self, iters = 1000, alpha=0.1):
        '''

        :param iters: ��������
        :param alpha: ����
        :return:
        '''

        # �����ʼ��Ȩ�ؾ���
        self.weights = np.random.rand(self.n_class, self.n_features)

        # ���� one-hot ����(���ȱ���)
        y_one_hot = self.one_hot(self.train_y)


        # ����������ʧ���
        self.all_loss = []
        for i in range(iters):  # ��������
            scores = self.train_x * self.weights.T
            # ����softmax��ֵ(����������ĸ���)
            probs = self.softmax(scores)
            # ���ۺ���    np.multiply��Ӧλ����ˣ����
            loss = -(1.0 / self.n_samples) * np.sum(np.multiply(y_one_hot, np.log(probs)))
            self.all_loss.append(loss)
            # ����ݶ�
            grad = -(1.0 / self.n_samples) * ((y_one_hot - probs)).T * self.train_x
            # ����Ȩ�ؾ���
            self.weights = self.weights - alpha * grad

    #Ԥ������
    def predict(self, test_x):
        '''

        :param test_x: ���Լ�
        :return:
        '''
        scores = test_x * self.weights.T
        probs = self.softmax(scores)

        return np.argmax(probs, axis=1)  # ���ز��Լ�Ԥ������

    #ģ������
    def test(self, test_x, test_y):
        '''

        :param test_x: ���Լ�
        :param test_y: �Ӳ��Լ����
        :return:
        '''

        y_predict = self.predict(test_x)
        accuray = np.sum(y_predict == test_y) / len(test_y)

        print("���Լ���ȷ�ʣ�%.4f" % accuray)

        # ������ʧ����
        plt.figure()
        plt.title("loss - train iter")
        plt.xlabel("train iter")
        plt.ylabel("loss")
        plt.plot(np.arange(1000), self.all_loss)
        plt.show()


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #������
    target = iris_data['target']  #��������
    data = iris_data['data']  #���ݼ�
    target = target.reshape(150, 1)
    train_x, test_x,  train_y, test_y = train_test_split(data, target, test_size = 0.2, random_state = 7)

    logistic = Logistic(train_x, train_y)
    logistic.fit()
    logistic.test(test_x, test_y)