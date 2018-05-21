import re
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from my_pearson import my_corr

zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')


# 这个写的非常好！
def bag_of_word(list_names):

    matrix_names = []
    for n in list_names:
        # print(n)
        rst = jieba.cut(n)
        rst = ' '.join([r for r in rst])
        matrix_names.append(rst)
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(matrix_names)
    # print(X_train_counts)
    print(X_train_counts.shape)
    # print(count_vect.vocabulary_)
    tf_transformer = TfidfTransformer(norm='l2', use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    print(type(X_train_tf))
    return X_train_tf


def get_target(type):
    data = pd.read_csv('data/psy_test.txt', sep='\t')
    d = data['Extraversion']
    if type == 'value':
        return d
    elif type == 'class':
        rst = []
        for v in d:
            if v > 42.81:
                rst.append(1)
            elif v > 35.25:
                rst.append(0)
            else:
                rst.append(-1)
        print(rst)
        return rst


def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    global zh_pattern
    match = zh_pattern.search(word)

    return match


def get_nickname_features(name):
    # 长度
    _len = len(name)
    # 中文比例
    len_chi = 0
    for w in name:
        if contain_zh(w):
            len_chi += 1
    return _len, len_chi / _len


if __name__ == '__main__':
    list_names = []
    x1 = []
    x2 = []
    for line in open('data/nickname.txt'):
        _id, name = line.strip().split(' ')
        print(name)
        _x1, _x2 = get_nickname_features(name)
        list_names.append(name)
        x1.append(_x1)
        x2.append(_x2)

    y = get_target('value') 
    my_corr(x1, y)
    my_corr(x2, y)


    # 训练模型

    X = bag_of_word(list_names)
    y = get_target('class')
    clf = LinearSVC(random_state=0)
    print(pd.Series(cross_val_score(clf, X, y, cv=5)).mean())