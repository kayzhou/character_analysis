# -*- coding: utf-8 -*-
__author__ = 'Kay'

from feature_handler import *
from sklearn.feature_extraction.text import CountVectorizer
import json
from NLP_tool import appear_words_voc


def str2datetime(s):
    try:
        re = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except:
        re = datetime.datetime.strptime(s[:-11] + s[-5:], '%c')
    return re


def get_dynamic_features(in_name):
    return file_dynamic_features(in_name)


def get_text_features(all_text):
    # corpus 必须是数组
    # corpus = [seg_word(filter(all_text))]
    vector = CountVectorizer(vocabulary=read_keyword_set(), binary=True)
    try:
        re = vector.fit_transform(all_text).toarray()
    except:
        print(all_text)
    x = [str(r) for r in re[0]]
    return x


def get_shopping_text_features(all_text):
    # corpus 必须是数组
    # corpus = [seg_word(filter(all_text))]
    # vector = CountVectorizer(vocabulary=['买', '购物', '购买', '淘宝', '京东', 'buy', 'shop'])
    vector = CountVectorizer(vocabulary=['买'])

    x = vector.fit_transform(all_text).toarray()
    print(x)
    print(vector.get_feature_names())

    return x


def get_features(in_name):
    # 分类器所需特征
    one_line = open(in_name).readline().strip()
    last_dt = last_weibo(in_name)
    all_text = ""
    for line in open(in_name, encoding='utf8'):
        json_data = json.loads(line.strip())
        all_text += json_data["text"]

    return line_static_features(one_line, last_dt) + get_dynamic_features(in_name) + appear_words_voc(all_text, read_keyword_set())

    # 新的文本特征
    # cnt = how_many_weibo(in_name)
    # if cnt < 100:
    #     return None
    # all_text = [""]
    # for line in open(in_name, encoding='utf8'):
    #     json_data = json.loads(line.strip())
    #     all_text[0] += json_data["text"]
    # x = get_shopping_text_features(all_text)
    # return x[0] / cnt


if __name__ == '__main__':

    dir_name = '../extract_weibo_users/weibo_0320'
    dir_name = "/Users/Kay/Project/EXP/character_analysis/data/users_20160302"
    out_file = open('features_large_IGNORE_311.txt', 'w')
    for in_name in os.listdir(dir_name):
        if len(in_name) != 10: # 长度为10是有效的uid
            continue
        elif how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
            continue

        print(in_name)
        X = get_features(dir_name + "/" + in_name)
        if X:
            out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")

    # out_file = open('features_328_shopping.txt', 'w')
    # for in_name in os.listdir('../extract_weibo_users/weibo_0320'):
    #     if len(in_name) == 10: # 为有效的uid
    #         print(in_name)
    #         X = get_features("../extract_weibo_users/weibo_0320/" + in_name)
    #         if X.any():
    #             out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")

    # get_shopping_text_features(['买'])