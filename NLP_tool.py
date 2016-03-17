# -*- coding: utf-8 -*-
__author__ = 'Kay'

import jieba
import jieba.posseg as pseg
import emotion_cla.filer
import emotion_cla.separate
import codecs
jieba.load_userdict('NLP_data/user_dict.txt')
stop_word_set = set([word.strip() for word in codecs.open('NLP_data/stop_word.txt').readlines()])
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np


def filter(s):
    '''
    过滤文本
    :param s:
    :return:
    '''
    return emotion_cla.filer.filter(s)


def seg_word(s):
    '''
    分词并进行基本过滤, 过滤停用词
    :param s:
    :return:
    '''
    word_list=[word for word in jieba.cut(s) if word and not word.isdigit() and not word in stop_word_set]
    return ' '.join(word_list)


def seg_word_class(s):
    '''
    分词并进行词性标注
    :param s:
    :return:
    '''
    seg_list = pseg.cut(s)
    return ' '.join([str(w.word) for w in seg_list if w.flag != 'x'])
    # return ' '.join([str(w) for w in seg_list])


def delete_blank(s):
    '''
    删除多余空格
    :param s:
    :return:
    '''
    return ' '.join(s.split(' '))


def tf_idf(in_dir, out_dir):
    '''
    载入文本(一行), 并计算tfidf写入文件
    :param in_dir:
    :param out_dir:
    :return:
    '''

    def read_keyword_set():
        return set([line.strip() for line in open('words_related.txt', encoding='utf8')])

    corpus = []
    files = []
    for file_name in os.listdir(in_dir):
        if os.path.isfile(os.path.join(in_dir, file_name)) and file_name.endswith('.txt'):
            print(file_name)
            # print(os.path.join(in_dir, file_name).read())
            files.append(file_name)
            corpus.append(open(os.path.join(in_dir, file_name), encoding='utf8').read())

    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(min_df=5, vocabulary=read_keyword_set())
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight = tf_idf.toarray()

    # 打印词表
    out_file = open('data/word_list.txt', 'a', encoding='utf8')
    for i in range(len(word)):
        out_file.write(str(word[i]) + '\n')
    out_file.close()

    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        print("------ 第", i ,u"类文本的词语tf-idf权重 ------")
        for j in range(len(word)):
            open(out_dir + '/' + files[i], 'a', encoding='utf8').write(str(weight[i][j])+ ' ')


def file2vector(in_name):
    print(in_name)
    return np.loadtxt(in_name)


def count_words(in_name, out_name):
    dic={}
    for i in open(in_name, encoding='utf8'):
        i=i.strip()
        array=i.split(' ')
        for j in array:
            if j not in dic:
                dic[j]=0
            dic[j]+=1

    f=open(out_name, 'w', encoding='utf8')
    dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    # print(dic)
    for i in dic:
        if i[1] > 1000 or i[1] < 100 \
                or len(i[0]) == 1 \
                    or emotion_cla.separate.is_stop_word(i[0]): continue
        # f.write(i[0] + '\t' + str(i[1]) +'\n')
        f.write(i[0] + '\n')
    f.close()


def count_words_voc(in_name, out_name, voc=True):
    def read_keyword_set():
        return set([line.strip() for line in open('data/keyword.txt', encoding='utf8')])

    dic = {}
    for kw in read_keyword_set():
        dic[kw] = 0

    for i in open(in_name, encoding='utf8'):
        i=i.strip()
        array=i.split(' ')
        for j in array:
            if j in dic:
                dic[j]+=1

    f=open(out_name, 'w', encoding='utf8')
    dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    # print(dic)
    for i in dic:
        if i[1] > 1000 or i[1] < 10 \
                or len(i[0]) == 1 \
                    or emotion_cla.separate.is_stop_word(i[0]): continue
        # f.write(i[0] + '\t' + str(i[1]) +'\n')
        f.write(i[0] + '\n')
    f.close()


def vectorizer(in_name, out_name):
    '''
    词频统计
    :param in_name:
    :param out_name:
    :return:
    '''
    def read_keyword_set():
        return set([line.strip() for line in open('data/keyword.txt', encoding='utf8')])

    corpus = [line.strip() for line in open(in_name, encoding='utf8')]
    # vector = CountVectorizer(vocabulary=read_keyword_set())
    vector = CountVectorizer()
    re = vector.fit_transform(corpus)
    print(re.toarray().shape)
    word = vector.get_feature_names()

    # 打印词表
    out_file = open('data/feature_word.txt', 'a', encoding='utf8')
    for i in range(len(word)):
        out_file.write(str(word[i]) + '\n')
    out_file.close()
    # print(re.toarray().sum())


def vectorizer_dir(in_dir, out_name, voc_name):
    '''
    向量化
    :param in_name:
    :param out_name:
    :return:
    '''
    def read_keyword_set():
        return set([line.strip() for line in open(voc_name, encoding='utf8')])

    corpus = []
    for file_name in os.listdir(in_dir):
        if os.path.isfile(os.path.join(in_dir, file_name)) and file_name.endswith('.txt'):
            print(file_name)
            # print(os.path.join(in_dir, file_name).read())
            corpus.append(open(os.path.join(in_dir, file_name), encoding='utf8').read())


    vector = CountVectorizer(vocabulary=read_keyword_set())
    re = vector.fit_transform(corpus).toarray()
    out_file = open(out_name, 'w')
    for row in re:
        out_file.write(' '.join([str(int(bool(r))) for r in row]) + '\n')
        # out_file.write(' '.join([str(r) for r in row]) + '\n')


if __name__ == '__main__':
    # print(seg_word(filter('//转发微博.')))
    # tf_idf('data/text_one_line', 'data/tfidf_scale')
    # count_words_voc('data/text_data.txt', 'keyword.txt')
    vectorizer_dir('data/text_one_line', 'data/word_appear_scale.txt', 'data/keyword.txt')