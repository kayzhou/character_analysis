# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn import tree
from numpy import loadtxt
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def ca_tree(in_name, out_model_name):
    # print(in_name)
    X, y = load_svmlight_file(in_name)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    cvs = 0
    for i in range(10):
        cvs += cross_val_score(clf, X, y, cv=10).mean()
    print('cross_val_score =', cvs / 10)
    y_hat = clf.predict(X)
    # print('预测结果 =', y_hat); print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    # print('score =', clf.score(X, y))
    print('F1 score =', f1_score(y, y_hat))


def ca_rf(in_name, out_model_name):
    # print(in_name)
    X, y = load_svmlight_file(in_name)
    clf = RandomForestClassifier()
    clf = clf.fit(X, y)
    cvs = 0
    for i in range(10):
        cvs += cross_val_score(clf, X, y, cv=10).mean()
    print('cross_val_score =', cvs / 10)
    y_hat = clf.predict(X)
    print('预测结果 =', y_hat)
    # print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    # print('score =', clf.score(X, y))
    print('F1 score =', f1_score(y, y_hat, average=None))


if __name__ == '__main__':

    ca_tree('data/SVM/328_IGNORE_features_1.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/328_IGNORE_features_sides_0.txt', 'model/rf_311_0.mod')
    # ca_rf('data/SVM/328_IGNORE_features_sides_1.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/328_IGNORE_features_sides_2.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/328_IGNORE_features_sides_3.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/328_IGNORE_features_sides_4.txt', 'model/rf_311_4.mod')

    # ca_rf('data/SVM/328_IGNORE_features_0.txt', 'model/rf_311_0.mod')
    ca_rf('data/SVM/328_IGNORE_features_1.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/328_IGNORE_features_2.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/328_IGNORE_features_3.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/328_IGNORE_features_4.txt', 'model/rf_311_4.mod')

