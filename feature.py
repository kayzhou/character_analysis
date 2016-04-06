# -*- coding: utf-8 -*-
__author__ = 'Kay'

from feature_handler import *
from sklearn.feature_extraction.text import CountVectorizer
import json
from NLP_tool import appear_words_voc, count_words_voc


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
    def read_keyword_list():
        return list([line.strip() for line in open('NLP_data/shopping_behaviour_word.txt', encoding='utf8')])
    return count_words_voc(all_text, read_keyword_list())


def get_train_features(in_name):
    # 分类器所需特征

    # ! 前方高能预警 !
    # --- !!! order 很重要, 必填 !!! ---
    # 你问我为什么??!
    # 因为我要取到最近的一条微博啊! 不同的获取方式微博排序顺序不同, 第一个? 还是最后一个? 是最新的, I don't KNOW!
    # ! 前方高能预警 !

    last_w = last_weibo(in_name, order=False)
    line = json.dumps(last_w)
    all_text = ""
    for line in open(in_name, encoding='utf8'):
        json_data = json.loads(line.strip())
        all_text += json_data["text"]

    return line_static_features(line, str2datetime(last_w['created_at'])) \
           + get_dynamic_features(in_name) \
           + appear_words_voc(all_text, read_keyword_set())


def get_behavior_features(in_name):

    # 新的文本特征
    cnt = how_many_weibo(in_name)
    x = [cnt] # 第一个特征为爬取的微博个数
    all_text = []
    for line in open(in_name, encoding='utf8').readlines():
        all_text.append(line.strip())

    word_cnt = get_shopping_text_features(" ".join(all_text))
    x += (word_cnt).tolist()
    print(x)
    return x


def get_badge_features(in_name):
    # order is fatal
    last_w = last_weibo(in_name, order=False)
    line = json.dumps(last_w)
    return line_get_badge(line)


def get_mood_features(in_name):
    return extract_mood(in_name)


if __name__ == '__main__':

    # 提取训练数据特征
    # dir_name = '../extract_weibo_users/weibo_0320'
    # dir_name = 'data/tmp'
    dir_name = "/Users/Kay/Project/EXP/character_analysis/data/users_20160302"
    # out_file = open('train_IGNORE_404.txt', 'w')
    # for in_name in os.listdir(dir_name):
    #     if len(in_name) != 10: # 长度为10是有效的uid
    #         continue
    #     elif how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
    #         continue
    #
    #     print(in_name)
    #     X = get_train_features(dir_name + "/" + in_name)
    #     if X:
    #         out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")


    # 提取需要分析的文本特征
    # out_file = open('large_401_shopping.txt', 'w')
    # for in_name in os.listdir(dir_name):
    #     if len(in_name) != 10: # 长度为10是有效的uid
    #         continue
    #     elif how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
    #         continue
    #     print(in_name)
    #     X = get_behavior_features(dir_name + "/" + in_name)
    #     out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")


    # 提取徽章信息
    # out_file = open('train_401_badge.txt', 'w')
    # for in_name in os.listdir(dir_name):
    #     if len(in_name) != 10: # 长度为10是有效的uid
    #         continue
    #     elif how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
    #         continue
    #     print(in_name)
    #     X = get_badge_features(dir_name + "/" + in_name)
    #     out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")


    # 提取情绪信息
    out_file = open('train_406_mood.txt', 'w')
    for in_name in os.listdir(dir_name):
        if len(in_name) != 10: # 长度为10是有效的uid
            continue
        elif how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
            continue
        print(in_name)
        X = get_mood_features(dir_name + "/" + in_name)
        out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")