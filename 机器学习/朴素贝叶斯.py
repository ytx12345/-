import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Gaussian():
    def __init__(self):

        self.priors = None  #�������
        self.means = None  #��ֵ
        self.vars = None  #����

    # ��1����ѵ�������������������
    def prior_predict(self, train_y):
        '''

        :param train_y: ѵ����
        :return:
        '''
        prior = []  #�������������
        classify = set(train_y) #��� {0,1,2}
        n_samples = len(train_y)  #������

        for label in classify:
            prior.append(np.sum(train_y[:] == label) / n_samples)

        return prior

    #��2����������������ͷ���
    def mean_cov(self, train_x, train_y):
        '''

        :param train_x: ѵ����
        :param train_y: ѵ������Ӧ�����
        :return:
        '''
        mean = [] #����
        cov_matrix = [] #����

        for label in set(train_y):
            cov = np.zeros((4,4))
            label_data = train_x[train_y == label]
            mean.append(np.mean(label_data, axis=0)) #��ÿ������ÿ����������
            (cov[0, 0], cov[1, 1], cov[2, 2], cov[3, 3]) = np.var(label_data, axis=0)
            cov_matrix.append(cov) #��ÿ�����ÿ�����Է���

        return mean, cov_matrix

    #��3����������������
    def class_probability(self, x, mean, cov):
        '''

        :param x: ��Ԥ������
        :param mean: ��ֵ
        :param cov: ����
        :return:
        '''
        x1 = np.mat(x).T  # ����������
        mean1 = np.mat(mean).T  # ����
        cov1 = np.mat(cov)  # Э����

        m = -((x1 - mean1).T * cov1.I * (x1 - mean1)) / 2  # ���������ʺ�����ָ����
        n = 2 * 3.14 * (np.linalg.det(cov1) ** 1 / 2)  # ���������ʵĵĳ�������
        p = (1 / n) * np.math.exp(np.linalg.det(m))  # ���������ʵ�Ԥ��ֵ

        return p
    #��4��Ԥ�����
    def predict(self, x):
        '''

        :param x: ��Ԥ������
        :return:
        '''
        labels = []
        for mean, cov, prior in zip(self.means, self.vars, self.priors):
            pre = self.class_probability(x, mean, cov)
            pre = pre * prior
            labels.append(pre)

        return np.argmax(labels)

    #ģ������
    def fit(self, train_x, train_y):
        '''

        :param train_x: ѵ����
        :param train_y: ѵ������Ӧ�����
        :return:
        '''

        self.priors = self.prior_predict(train_y) #�������
        self.means, self.vars = self.mean_cov(train_x, train_y) #����������

    #ģ�͵�����
    def test(self, test_x, test_y):
        '''

        :param test_x: ���Լ�
        :param test_y: ���Լ���Ӧ�����
        :return:
        '''

        predicts = []  #Ԥ�����

        for x in test_x:
            predicts.append(self.predict(x))

        accuray = np.sum(predicts == test_y) / len(test_y)  #��ȷ��

        print("���Լ���ȷ�ʣ�%.4f" % accuray)


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #������
    target = iris_data['target']  #��������
    data = iris_data['data']  #���ݼ�

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.1, random_state = 0)

    gaussian = Gaussian()
    gaussian.fit(train_x, train_y)
    gaussian.test(test_x, test_y)