# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
import numpy as np

'''
grid 最优参数

328
SVM 三分类：
8.0 3.0517578125e-05 40.273
8.0 0.0078125 39.5904
32.0 0.00048828125 39.5904
8.0 0.0001220703125 48.4642
2.0 0.001953125 45.0512

SVM 二分类：
2.0 0.0001220703125 54.7872
8192.0 3.0517578125e-05 59.375
8192.0 0.0001220703125 54.4503
32.0 0.001953125 58.4416
2.0 0.0001220703125 57.6471

SVM 二分类：
2.0 0.03125 52.3438
512.0 3.0517578125e-05 60.4839
128.0 3.0517578125e-05 53.2258
2048.0 3.0517578125e-05 61.7647
2.0 0.0078125 64.2202

0.5 0.00048828125 41.3793
2.0 0.0078125 46.8966
0.5 0.00048828125 36.5517
0.03125 0.0078125 49.6552
2.0 0.0001220703125 46.2069
'''

def ca_svm(in_name, out_model_name, C, gamma):
    print(in_name)
    X, y = load_svmlight_file(in_name)
    clf = SVC(C=C, gamma=gamma, probability=True)
    clf.fit(X, y)
    cvs = cross_val_score(clf, X, y, cv=10).mean()
    print('10次10折交叉检验 =', cvs)
    y_hat = clf.predict(X)
    print('预测结果 =', y_hat)
    print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    print('训练数据上的表现 =', clf.score(X, y))
    # print('F1 score =', f1_score(y, y_hat))
    print('F1 score =', f1_score(y, y_hat, average='macro'))
    print('----------------------------------------------')



def ca_svm_grid(in_name, out_model_name):
    '''
    自己选择最好的参数
    :param in_name:
    :param out_model_name:
    :return:
    '''
    print(in_name)
    X, y = load_svmlight_file(in_name)
    # param_grid = {'C': np.logspace(-10, 10, num=21, base=2),
    #               'gamma': np.logspace(-10, 10, num=21, base=2)}
    param_grid = {'C': np.logspace(-5, 5, num=11, base=2),
                  'gamma': np.logspace(-5, 5, num=11, base=2)}
    clf = SVC(probability=True)
    # clf = GridSearchCV(clf, param_grid, scoring='f1', cv=5)
    # clf = GridSearchCV(clf, param_grid, scoring='log_loss', cv=5)
    clf = GridSearchCV(clf, param_grid, cv=10)
    clf.fit(X, y)
    print('最优交叉验证结果:', clf.best_score_)
    print('最优参数:', clf.best_params_)

    y_hat = clf.predict(X)
    print('预测结果 =', y_hat)
    # print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    # print('F1 score =', f1_score(y, y_hat, average='macro'))
    print('F1 score =', f1_score(y, y_hat))
    print('训练数据上的表现 =', clf.score(X, y))


def svm_predict(in_name, model):
    clf = joblib.load(model)
    for line in open(in_name):
        X = np.array([[float(x) for x in line.strip()[11:].split(' ')]])
        y = clf.predict(X)
        # y_pro = clf.predict_proba(X)
        # print(line[:10], y, y_pro)
        # if y[0] == -1:
        #     print(line[:10], y_pro[0][0], y_pro[0][1], y_pro[0][2])
        if y[0] != 0:
            print(line[:10], y[0])


if __name__ == '__main__':

    # ca_svm('data/SVM/404_features_0.txt', 'model/svm_404_features_0.mod', 0.5, 0.00048828125)
    # ca_svm('data/SVM/404_features_1.txt', 'model/svm_404_features_1.mod', 2.0, 0.0078125)
    # ca_svm('data/SVM/404_features_2.txt', 'model/svm_404_features_2.mod', 0.5, 0.00048828125)
    # ca_svm('data/SVM/404_features_3.txt', 'model/svm_404_features_3.mod', 0.03125, 0.0078125)
    # ca_svm('data/SVM/404_features_4.txt', 'model/svm_404_features_4.mod', 2.0, 0.0001220703125)
    
    # ca_svm('data/SVM/404_NOR_features_0.txt', 'model/svm_404_NOR_features_0.mod', 0.5, 0.00048828125)
    ca_svm('data/SVM/404_NOR_features_1.txt', 'model/svm_404_NOR_features_1.mod', 512.0, 0.0001220703125)
    # ca_svm('data/SVM/404_NOR_features_2.txt', 'model/svm_404_NOR_features_2.mod', 0.5, 0.00048828125)
    # ca_svm('data/SVM/404_NOR_features_3.txt', 'model/svm_404_NOR_features_3.mod', 0.03125, 0.0078125)
    # ca_svm('data/SVM/404_NOR_features_4.txt', 'model/svm_404_NOR_features_4.mod', 2.0, 0.0001220703125)

    # ca_svm('data/SVM/404_features_sides_1.txt', 'model/svm_404_NOR_features_3.mod', 32.0, 0.001953125)
    # ca_svm('data/SVM/404_NOR_features_sides_2.txt', 'model/svm_404_NOR_features_4.mod', 512.0, 0.0001220703125)


    # ca_svm_grid('data/SVM/328_IGNORE_features_0.txt', 'model/svm_328_features_0.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_1.txt', 'model/svm_328_features_1.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_2.txt', 'model/svm_328_features_2.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_3.txt', 'model/svm_328_features_3.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_4.txt', 'model/svm_328_features_4.mod')

    # ca_svm_grid('data/SVM/328_IGNORE_features_sides_0.txt', 'model/svm_328_features_sides_0.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_sides_1.txt', 'model/svm_328_features_sides_1.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_sides_2.txt', 'model/svm_328_features_sides_2.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_sides_3.txt', 'model/svm_328_features_sides_3.mod')
    # ca_svm_grid('data/SVM/328_IGNORE_features_sides_4.txt', 'model/svm_328_features_sides_4.mod')

    # svm_predict('data/features/large_IGNORE_404_NOR.txt', 'model/svm_404_NOR_features_1.mod')

    # svm_predict('data/features/large_IGNORE_404.txt', 'model/svm_404_features_4.mod')