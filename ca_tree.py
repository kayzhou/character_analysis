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
    # print('预测结果 =', y_hat); print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    # print('score =', clf.score(X, y))
    print('F1 score =', f1_score(y, y_hat, average=None))


if __name__ == '__main__':
    # print('static')
    # ca_tree('data/train/20160304_static_0_class.txt', 'model/tree_304_static_0.mod')
    # ca_tree('data/train/20160304_static_1_class.txt', 'model/tree_304_static_1.mod')
    # ca_tree('data/train/20160304_static_2_class.txt', 'model/tree_304_static_2.mod')
    # ca_tree('data/train/20160304_static_3_class.txt', 'model/tree_304_static_3.mod')
    # ca_tree('data/train/20160304_static_4_class.txt', 'model/tree_304_static_4.mod')
    # print('dynamic')
    # ca_tree('data/SVM/306_dynamic_0_class.txt', 'model/tree_306_dynamic_0.mod')
    # ca_tree('data/SVM/306_dynamic_1_class.txt', 'model/tree_306_dynamic_1.mod')
    # ca_tree('data/SVM/306_dynamic_2_class.txt', 'model/tree_306_dynamic_2.mod')
    # ca_tree('data/SVM/306_dynamic_3_class.txt', 'model/tree_306_dynamic_3.mod')
    # ca_tree('data/SVM/306_dynamic_4_class.txt', 'model/tree_306_dynamic_4.mod')
    # print('tfidf')
    # ca_tree('data/SVM/306_tfidf_min5_0.txt', 'model/tree_306_tfidf_min5_0.mod')
    # ca_tree('data/SVM/306_tfidf_min5_1.txt', 'model/tree_306_tfidf_min5_1.mod')
    # ca_tree('data/SVM/306_tfidf_min5_2.txt', 'model/tree_306_tfidf_min5_2.mod')
    # ca_tree('data/SVM/306_tfidf_min5_3.txt', 'model/tree_306_tfidf_min5_3.mod')
    # ca_tree('data/SVM/306_tfidf_min5_4.txt', 'model/tree_306_tfidf_min5_4.mod')


    # print('static')
    # ca_rf('data/train/20160304_static_0_class.txt', 'model/rf_304_static_0.mod')
    # ca_rf('data/train/20160304_static_1_class.txt', 'model/rf_304_static_1.mod')
    # ca_rf('data/train/20160304_static_2_class.txt', 'model/rf_304_static_2.mod')
    # ca_rf('data/train/20160304_static_3_class.txt', 'model/rf_304_static_3.mod')
    # ca_rf('data/train/20160304_static_4_class.txt', 'model/rf_304_static_4.mod')
    # print('dynamic')
    # ca_rf('data/SVM/306_dynamic_0_class.txt', 'model/rf_306_dynamic_0.mod')
    # ca_rf('data/SVM/306_dynamic_1_class.txt', 'model/rf_306_dynamic_1.mod')
    # ca_rf('data/SVM/306_dynamic_2_class.txt', 'model/rf_306_dynamic_2.mod')
    # ca_rf('data/SVM/306_dynamic_3_class.txt', 'model/rf_306_dynamic_3.mod')
    # ca_rf('data/SVM/306_dynamic_4_class.txt', 'model/rf_306_dynamic_4.mod')
    # print('tfidf')
    # ca_rf('data/SVM/306_tfidf_min5_0.txt', 'model/rf_306_tfidf_min5_0.mod')
    # ca_rf('data/SVM/306_tfidf_min5_1.txt', 'model/rf_306_tfidf_min5_1.mod')
    # ca_rf('data/SVM/306_tfidf_min5_2.txt', 'model/rf_306_tfidf_min5_2.mod')
    # ca_rf('data/SVM/306_tfidf_min5_3.txt', 'model/rf_306_tfidf_min5_3.mod')
    # ca_rf('data/SVM/306_tfidf_min5_4.txt', 'model/rf_306_tfidf_min5_4.mod')


    # print('static')
    # ca_rf('data/SVM/311_static_0_class_side.txt', 'model/rf_311_static_0.mod')
    # ca_rf('data/SVM/311_static_1_class_side.txt', 'model/rf_311_static_1.mod')
    # ca_rf('data/SVM/311_static_2_class_side.txt', 'model/rf_311_static_2.mod')
    # ca_rf('data/SVM/311_static_3_class_side.txt', 'model/rf_311_static_3.mod')
    # ca_rf('data/SVM/311_static_4_class_side.txt', 'model/rf_311_static_4.mod')

    # ca_rf('data/SVM/311_static_0_class.txt', 'model/rf_311_static_0.mod')
    # ca_rf('data/SVM/311_static_1_class.txt', 'model/rf_311_static_1.mod')
    # ca_rf('data/SVM/311_static_2_class.txt', 'model/rf_311_static_2.mod')
    # ca_rf('data/SVM/311_static_3_class.txt', 'model/rf_311_static_3.mod')
    # ca_rf('data/SVM/311_static_4_class.txt', 'model/rf_311_static_4.mod')
    # print('dynamic')
    # ca_rf('data/SVM/311_dynamic_0_class.txt', 'model/rf_311_dynamic_0.mod')
    # ca_rf('data/SVM/311_dynamic_1_class.txt', 'model/rf_311_dynamic_1.mod')
    # ca_rf('data/SVM/311_dynamic_2_class.txt', 'model/rf_311_dynamic_2.mod')
    # ca_rf('data/SVM/311_dynamic_3_class.txt', 'model/rf_311_dynamic_3.mod')
    # ca_rf('data/SVM/311_dynamic_4_class.txt', 'model/rf_311_dynamic_4.mod')
    # print('tfidf')
    # ca_rf('data/SVM/306_tfidf_min5_0.txt', 'model/rf_306_tfidf_min5_0.mod')
    # ca_rf('data/SVM/306_tfidf_min5_1.txt', 'model/rf_306_tfidf_min5_1.mod')
    # ca_rf('data/SVM/306_tfidf_min5_2.txt', 'model/rf_306_tfidf_min5_2.mod')
    # ca_rf('data/SVM/306_tfidf_min5_3.txt', 'model/rf_306_tfidf_min5_3.mod')
    # ca_rf('data/SVM/306_tfidf_min5_4.txt', 'model/rf_306_tfidf_min5_4.mod')

    # ca_rf('data/SVM/311_0_class_side.txt', 'model/rf_311_0.mod')
    # ca_rf('data/SVM/311_1_class_side.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/311_2_class_side.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/311_3_class_side.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/311_4_class_side.txt', 'model/rf_311_4.mod')

    # ca_rf('data/SVM/315_features_0_class.txt', 'model/rf_311_0.mod')
    # ca_rf('data/SVM/315_features_1_class.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/315_features_2_class.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/315_features_3_class.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/315_features_4_class.txt', 'model/rf_311_4.mod')

    # ca_rf('data/SVM/315_features_0.txt', 'model/rf_311_0.mod')
    # ca_rf('data/SVM/315_features_1.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/315_features_2.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/315_features_3.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/315_features_4.txt', 'model/rf_311_4.mod')
    
    # ca_rf('data/SVM/315_features_0.txt', 'model/rf_311_0.mod')
    # ca_rf('data/SVM/315_features_1.txt', 'model/rf_311_1.mod')
    # ca_rf('data/SVM/315_features_2.txt', 'model/rf_311_2.mod')
    # ca_rf('data/SVM/315_features_3.txt', 'model/rf_311_3.mod')
    # ca_rf('data/SVM/315_features_4.txt', 'model/rf_311_4.mod')

    ca_rf('data/SVM/315_features_0_sides.txt', 'model/rf_311_0.mod')
    ca_rf('data/SVM/315_features_1_sides.txt', 'model/rf_311_1.mod')
    ca_rf('data/SVM/315_features_2_sides.txt', 'model/rf_311_2.mod')
    ca_rf('data/SVM/315_features_3_sides.txt', 'model/rf_311_3.mod')
    ca_rf('data/SVM/315_features_4_sides.txt', 'model/rf_311_4.mod')

