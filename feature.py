# -*- coding: utf-8 -*-
__author__ = 'Kay'

from feature_handler import *
from sklearn.feature_extraction.text import CountVectorizer
import json


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


def get_features(in_name):
    one_line = open(in_name).readline().strip()
    last_dt = last_weibo(in_name)
    all_text = [""]
    for line in open(in_name, encoding='utf8'):
        json_data = json.loads(line.strip())
        all_text[0] += json_data["text"]

    return line_static_features(one_line, last_dt) + get_dynamic_features(in_name) + get_text_features(all_text)


if __name__ == '__main__':

    out_file = open('features_328.txt', 'w')
    # print(get_text_features('我是一个正直而且自信的人.'))
    # print(get_features("1080807495"))
    for in_name in os.listdir('../extract_weibo_users/weibo_0320'):
        if len(in_name) == 10: # 为有效的uid
            print(in_name)
            X = get_features("../extract_weibo_users/weibo_0320/" + in_name)
            if X:
                out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")