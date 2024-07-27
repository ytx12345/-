from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
(1)LinearSVC 是线性分类支持向量机，不能使用核函数方法处理非线性分类问题。
   LinearSVC 算法与 SVC 算法在使用 ‘linear’ 核函数时的结果基本一致，但 LinearSVC 是基于 liblinear 库实现，计算速度更快。
（2）NuSVC 是线性分类支持向量机的具体实现

SVC 和 NuSVC 都可以使用核函数方法实现非线性分类，但参数设置有所区别。对于多类别分类问题，通过构造多个“one-versus-one”的二值分类器逐次分类
'''

def svm(train_x,train_y,test_x,test_y,):

    C = [0.1, 0.5, 1, 2, 3.5, 5, 7.5, 10, 20]  #惩罚参数
    for c in C:
        print('------------------------惩罚参数C为%.2f-------------------'%c)
        # 核函数['linear'：线性、'poly'：多项式、'rbf'高斯]
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        print('核函数      正确率')
        for i in kernel_list:
            svm = SVC(kernel=i,C=c)
            svm.fit(train_x,train_y)
            score = svm.score(test_x,test_y)
            print('%s    %5f'%(i,score))

if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #特征名
    target = iris_data['target']  #样本类型
    data = iris_data['data']  #数据集
    train_x, test_x,  train_y, test_y = train_test_split(data, target, test_size = 0.2, random_state = 7)

    svm(train_x,train_y,test_x,test_y)