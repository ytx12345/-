from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
(1)LinearSVC �����Է���֧��������������ʹ�ú˺���������������Է������⡣
   LinearSVC �㷨�� SVC �㷨��ʹ�� ��linear�� �˺���ʱ�Ľ������һ�£��� LinearSVC �ǻ��� liblinear ��ʵ�֣������ٶȸ��졣
��2��NuSVC �����Է���֧���������ľ���ʵ��

SVC �� NuSVC ������ʹ�ú˺�������ʵ�ַ����Է��࣬�����������������𡣶��ڶ����������⣬ͨ����������one-versus-one���Ķ�ֵ��������η���
'''

def svm(train_x,train_y,test_x,test_y,):

    C = [0.1, 0.5, 1, 2, 3.5, 5, 7.5, 10, 20]  #�ͷ�����
    for c in C:
        print('------------------------�ͷ�����CΪ%.2f-------------------'%c)
        # �˺���['linear'�����ԡ�'poly'������ʽ��'rbf'��˹]
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        print('�˺���      ��ȷ��')
        for i in kernel_list:
            svm = SVC(kernel=i,C=c)
            svm.fit(train_x,train_y)
            score = svm.score(test_x,test_y)
            print('%s    %5f'%(i,score))

if __name__ == "__main__":
    iris_data = load_iris()
    feature_names = iris_data['feature_names']  #������
    target = iris_data['target']  #��������
    data = iris_data['data']  #���ݼ�
    train_x, test_x,  train_y, test_y = train_test_split(data, target, test_size = 0.2, random_state = 7)

    svm(train_x,train_y,test_x,test_y)