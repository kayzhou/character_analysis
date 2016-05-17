# -*- coding: utf-8 -*-
__author__ = 'Kay'


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import f1_score


def ca_bayes(in_name, out_model_name):
    # print(in_name)
    X, y = load_svmlight_file(in_name)
    X = X.todense()
    # print(X, y)
    clf = GaussianNB()
    clf = MultinomialNB()
    clf.fit(X, y)
    cvs = 0
    for i in range(10):
        cvs += cross_val_score(clf, X, y, cv=10).mean()
    print('cross_val_score =', cvs / 10)
    y_hat = clf.predict(X)
    # print('预测结果 =', y_hat); print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    print('score =', clf.score(X, y))
    # print('F1 score =', f1_score(y, y_hat, average=None))
    print('F1 score =', f1_score(y, y_hat))


if __name__ == '__main__':
    # ca_bayes('data/SVM/311_static_0_class.txt', 'model/bayes_311_static_0.mod')
    # ca_bayes('data/SVM/311_static_1_class.txt', 'model/bayes_311_static_1.mod')
    # ca_bayes('data/SVM/311_static_2_class.txt', 'model/bayes_311_static_2.mod')
    # ca_bayes('data/SVM/311_static_3_class.txt', 'model/bayes_311_static_3.mod')
    # ca_bayes('data/SVM/311_static_4_class.txt', 'model/bayes_311_static_4.mod')
    
    # ca_bayes('data/SVM/311_0_class_side.txt', 'model/bayes_311_0.mod')
    # ca_bayes('data/SVM/311_1_class_side.txt', 'model/bayes_311_1.mod')
    # ca_bayes('data/SVM/311_2_class_side.txt', 'model/bayes_311_2.mod')
    # ca_bayes('data/SVM/311_3_class_side.txt', 'model/bayes_311_3.mod')
    # ca_bayes('data/SVM/311_4_class_side.txt', 'model/bayes_311_4.mod')

    # ca_bayes('data/SVM/315_features_0.txt', 'model/bayes_311_0.mod')
    # ca_bayes('data/SVM/315_features_1.txt', 'model/bayes_311_1.mod')
    # ca_bayes('data/SVM/315_features_2.txt', 'model/bayes_311_2.mod')
    # ca_bayes('data/SVM/315_features_3.txt', 'model/bayes_311_3.mod')
    # ca_bayes('data/SVM/315_features_4.txt', 'model/bayes_311_4.mod')

    # ca_bayes('data/SVM/315_features_0_sides.txt', 'model/bayes_311_0.mod')
    # ca_bayes('data/SVM/315_features_1_sides.txt', 'model/bayes_311_1.mod')
    # ca_bayes('data/SVM/315_features_2_sides.txt', 'model/bayes_311_2.mod')
    # ca_bayes('data/SVM/315_features_3_sides.txt', 'model/bayes_311_3.mod')
    # ca_bayes('data/SVM/315_features_4_sides.txt', 'model/bayes_311_4.mod')

    ca_bayes('data/SVM/404_NOR_features_1.txt', 'model/rf_311_1.mod')