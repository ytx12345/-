from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class Linear():
    def __init__(self):
        self.w = 1
        self.b = 1

    def fit(self, x, y):

        self.x = x
        self.y = y

        alpha = 0.001  # 学习率
        n = len(self.x) #样本个数
        for j in range(0, 10): #迭代10次
            loss = 0  # 损失
            for i in range(0,n):
                h = self.w * self.x[i] + self.b #预测值
                loss += (1/n) * (self.y[i] - h) * (self.y[i] - h) #损失
                grad = self.y[i] - h #梯度
                self.w += alpha * grad * self.x[i]
                self.b += alpha * grad

            print('第%d次迭代loss:%f' % (j+1, loss))
        print('W:%.4f  b:%.4f' % (self.w, self.b))

    def predict(self,x):

        return self.w * x + self.b

    def plot_data(self):
        y = self.x * self.w + self.b  #预测值
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y)  # 画散点图
        plt.plot(self.x, y, 'r') #模型学习到的直线
        ax.set_xlabel('sepal length (cm)')
        ax.set_ylabel('sepal width (cm)')
        plt.show()


if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #特征名
    target = iris_data['target']  #样本类型
    data = iris_data['data']  #数据集

    model = Linear()
    model.fit(data[:50, 0], data[:50, 1])
    model.plot_data()
