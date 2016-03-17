# -*- coding: utf-8 -*-
__author__ = 'Kay'

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import f1_score

'''
grid 最优参数

304,静态特征
0.5,3.0517578125e-05,55.2901
128.0,0.00048828125,53.5836
2048.0,0.001953125,59.0444
8.0,0.0001220703125,60.4096
0.5,0.0078125,54.9488

306,文本特征,min5
0.03125,0.0078125,52.901
32768.0,0.0001220703125,55.2901
8.0,2.0,56.314
512.0,0.03125,56.314
2.0,0.5,56.6553

306,动态特征
8.0,0.0001220703125,58.3618
32.0,0.00048828125,56.9966
8.0,0.5,56.6553
32.0,0.5,55.6314
2.0,2.0,54.9488

311,静态特征,三分类
0.5 0.0001220703125 38.9078
2.0 3.0517578125e-05 43.686
32.0 0.00048828125 39.9317
32.0 3.0517578125e-05 48.8055
0.5 3.0517578125e-05 42.3208

311,静态特征,二分类（两端）：
2.0 3.0517578125e-05 58.5106
2.0 3.0517578125e-05 59.375
2048.0 0.00048828125 56.0209
2.0 0.0001220703125 64.9351
8.0 3.0517578125e-05 52.3529

315
两类：
0.03125 0.0001220703125 60.6383
128.0 0.0001220703125 62.5
2.0 3.0517578125e-05 54.9738
0.5 0.00048828125 63.6364
128.0 3.0517578125e-05 59.4118
'''


def ca_svm(in_name, out_model_name, C, gamma):
    X, y = load_svmlight_file(in_name)
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X, y)
    cvs = 0
    for i in range(10):
        cvs += cross_val_score(clf, X, y, cv=10).mean()
    print('10次10折交叉检验 =', cvs / 10)
    y_hat = clf.predict(X)
    # print('预测结果 =', y_hat); print('实际结果 =', y)
    # joblib.dump(clf, out_model_name)
    print('训练数据上的表现 =', clf.score(X, y))
    try:
        print('F1 score =', f1_score(y, y_hat))
        pass
    except:
        pass


if __name__ == '__main__':
    
    # print('static')
    # ca_svm('data/train/20160304_static_0_class.txt', 'model/svm_304_static_0.mod', 0.5,3.0517578125e-05)
    # ca_svm('data/train/20160304_static_1_class.txt', 'model/svm_304_static_1.mod', 128.0,0.00048828125)
    # ca_svm('data/train/20160304_static_2_class.txt', 'model/svm_304_static_2.mod', 2048.0,0.001953125)
    # ca_svm('data/train/20160304_static_3_class.txt', 'model/svm_304_static_3.mod', 8.0,0.0001220703125)
    # ca_svm('data/train/20160304_static_4_class.txt', 'model/svm_304_static_4.mod', 0.5,0.0078125)
    # print('dynamic')
    # ca_svm('data/SVM/306_dynamic_0_class.txt', 'model/svm_306_dynamic_0.mod', 0.03125,0.0078125)
    # ca_svm('data/SVM/306_dynamic_1_class.txt', 'model/svm_306_dynamic_1.mod', 32768.0,0.0001220703125)
    # ca_svm('data/SVM/306_dynamic_2_class.txt', 'model/svm_306_dynamic_2.mod', 8.0,2.0)
    # ca_svm('data/SVM/306_dynamic_3_class.txt', 'model/svm_306_dynamic_3.mod', 512.0,0.03125)
    # ca_svm('data/SVM/306_dynamic_4_class.txt', 'model/svm_306_dynamic_4.mod', 2.0,0.5)
    # print('tfidf')
    # ca_svm('data/SVM/306_tfidf_min5_0.txt', 'model/svm_306_tfidf_min5_0.mod', 8.0,0.0001220703125)
    # ca_svm('data/SVM/306_tfidf_min5_1.txt', 'model/svm_306_tfidf_min5_1.mod', 32.0,0.00048828125)
    # ca_svm('data/SVM/306_tfidf_min5_2.txt', 'model/svm_306_tfidf_min5_2.mod', 8.0,0.5)
    # ca_svm('data/SVM/306_tfidf_min5_3.txt', 'model/svm_306_tfidf_min5_3.mod', 32.0,0.5)
    # ca_svm('data/SVM/306_tfidf_min5_4.txt', 'model/svm_306_tfidf_min5_4.mod', 2.0,2.0)
    
    # print('static')
    # ca_svm('data/SVM/311_static_0_class.txt', 'model/svm_311_static_0.mod', 0.5, 0.0001220703125)
    # ca_svm('data/SVM/311_static_1_class.txt', 'model/svm_311_static_1.mod', 2.0, 3.0517578125e-05)
    # ca_svm('data/SVM/311_static_2_class.txt', 'model/svm_311_static_2.mod', 32.0, 0.00048828125)
    # ca_svm('data/SVM/311_static_3_class.txt', 'model/svm_311_static_3.mod', 32.0, 3.0517578125e-05)
    # ca_svm('data/SVM/311_static_4_class.txt', 'model/svm_311_static_4.mod', 0.5, 3.0517578125e-05)
    # print('static')
    # ca_svm('data/SVM/311_static_0_class_side.txt', 'model/svm_311_static_0.mod', 2.0, 3.0517578125e-05)
    # ca_svm('data/SVM/311_static_1_class_side.txt', 'model/svm_311_static_1.mod', 2.0, 3.0517578125e-05)
    # ca_svm('data/SVM/311_static_2_class_side.txt', 'model/svm_311_static_2.mod', 2048.0, 0.00048828125)
    # ca_svm('data/SVM/311_static_3_class_side.txt', 'model/svm_311_static_3.mod', 2.0, 0.0001220703125)
    # ca_svm('data/SVM/311_static_4_class_side.txt', 'model/svm_311_static_4.mod', 8.0, 3.0517578125e-05)


    ca_svm('data/SVM/315_features_0_sides.txt', 'model/svm_311_static_0.mod', 0.03125, 0.0001220703125)
    ca_svm('data/SVM/315_features_1_sides.txt', 'model/svm_311_static_1.mod', 128.0, 0.0001220703125)
    ca_svm('data/SVM/315_features_2_sides.txt', 'model/svm_311_static_2.mod', 2.0, 3.0517578125e-05)
    ca_svm('data/SVM/315_features_3_sides.txt', 'model/svm_311_static_3.mod', 0.5, 0.00048828125 )
    ca_svm('data/SVM/315_features_4_sides.txt', 'model/svm_311_static_4.mod', 128.0, 3.0517578125e-05)