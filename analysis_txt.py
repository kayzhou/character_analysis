# coding: utf8

import os
import json
import jieba
import numpy as np
import math
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def count_lines(in_dir):
    count = 0

    for i in range(1, 294):
        in_name = os.path.join(in_dir, str(i) + '.txt')
        for line in open(in_name):
            count += 1
    print(count)


def get_vec(in_dir):
    matrix = []
    keywords = read_keywords('data/keywords_ext_-1.txt')
    for i in range(1, 294):
        in_name = os.path.join(in_dir, str(i) + '.txt')
        vec = [0] * len(keywords)
        for line in open(in_name):
            line = json.loads(line.strip())
            txt = line['text']
            # print(txt)
            words = jieba.cut(txt)
            for w in words:
                try:
                    bingo = keywords.index(w)
                    vec[bingo] += 1
                except:
                    pass
        matrix.append(vec)
        print(vec, file=open('data/words_vec.txt', 'a'))
    # matrix = np.array(matrix)
    # M = TfidfTransformer()
    # result = M.fit_transform(M)
    # print(result)
    

def tfidf(in_name):
    tfidf = []
    matrix = []
    for line in open(in_name):
        vec = eval(line.strip())
        matrix.append(vec)
    m = np.array(matrix)

    for i in range(10000):
        tf = sum(m[:, i])
        idf = 0
        for j in range(293):
            if m[j, i] > 0:
                idf += 1
        try:
            idf = math.log(293 / idf)
        except:
            idf = 0
        tfidf.append(tf * idf)
    tfidf = np.array(tfidf)
    print(tfidf.mean())


def read_keywords(in_name):
    words = []
    for i, line in enumerate(open(in_name)):
        if i > 10000:
            break
        words.append(line.strip().split('\t')[0])
    return words

count_lines('../paper-data/user-data')
# get_vec('../paper-data/user-data')
# tfidf('data/words_vec.txt')