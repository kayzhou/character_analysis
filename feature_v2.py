# -*- coding: utf-8 -*-
__author__ = 'Kay'

'''
抽取用户特征, 调用feature_handler.py内函数
'''

import datetime
import json
import timer

from sklearn.feature_extraction.text import CountVectorizer

from feature_handler import *
from NLP_tool import appear_one_word_voc


# def str2datetime(s):
#     try:
#         re = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
#     except:
#         re = datetime.datetime.strptime(s[:-11] + s[-5:], '%c')
#     return re


def get_dynamic_features(in_name):
    return file_dynamic_features(in_name)


def get_text_features(all_text):
    # corpus 必须是数组
    # corpus = [seg_word(filter(all_text))]
    vector = CountVectorizer(vocabulary=read_keyword_set(), binary=True)
    try:
        re = vector.fit_transform(all_text).toarray()
    except:
        print('except!')
        print(all_text)
    x = [str(r) for r in re[0]]
    return x


def get_train_features(in_name):
    # 分类器所需特征

    # ! 前方高能预警 !
    # --- !!! order 很重要, 必填 !!! ---
    # 你问我为什么??!
    # 因为我要取到最近的一条微博啊! 不同的获取方式微博排序顺序不同, 第一个? 还是最后一个? 是最新的, I don't KNOW!
    # order 为True则微博是按时间正序排列

    # 为什么会出现这样的问题?
    # 直接用api去获取是倒序的, 从Hbase取出来是正序的;
    # ! 前方高能预警 !

    last_w = last_weibo(in_name, order=False)
    last_line = json.dumps(last_w)

    return line_static_features(last_line, str2datetime(last_w['created_at'])) \
           + get_dynamic_features(in_name)


def get_behavior_features(in_name):

    # 新的文本特征
    cnt = how_many_weibo(in_name)
    x = [cnt] # 第一个特征为爬取的微博个数
    '''
    统计关键词的数量
    all_text = []
    for line in open(in_name, encoding='utf8').readlines():
        all_text.append(line.strip())

    word_cnt = get_shopping_text_features(" ".join(all_text))
    x += (word_cnt).tolist()
    print(x)
    return x
    '''

    def read_keyword_list():
        return list([line.strip() for line in open('NLP_data/shopping_behaviour_word.txt', encoding='utf8')])

    keyword_list = read_keyword_list()

    # 出现关键词则记为1，两个或多个关键词也是1！与上面区别开
    shopping_tweet = 0
    for line in open(in_name, encoding='utf8').readlines():
        line = json.loads(line)
        text = line['text']

        # 考虑转发的文本
        if 'retweeted_status' in line:
            text += line['retweeted_status']['text']

        if appear_one_word_voc(text.lower(), keyword_list):
            shopping_tweet += 1

    x.append(shopping_tweet)
    print(x)
    return x


def get_badge_features(in_name):
    # order is fatal
    last_w = last_weibo(in_name, order=True)
    line = json.dumps(last_w)
    return line_get_badge(line)


def get_mood_features(in_name):
    return extract_mood(in_name)


def get_keyword_features(in_name):
    seg = []
    for line in open(in_name, encoding='utf8').readlines():
        line = json.loads(line)
        seg += line['seg']
    return seg


def get_geo_features(in_name):
    geo = []
    for line in open(in_name, encoding='utf8').readlines():
        line = json.loads(line)
        if line['geo']:
            geo.append(str(line['geo']['coordinates'][0]) + ',' + str(line['geo']['coordinates'][1]))

    return geo


def get_url_features(in_name):
    url = 0
    for line in open(in_name, encoding='utf8').readlines():
        line = json.loads(line)
        if 'url_struct' in line:
            url += 1
    return [url]


def get_source_features(in_name):
    source = []
    for line in open(in_name, encoding='utf8').readlines():
        line = json.loads(line)
        if line['source']:
            source.append(line['source'])

    return source


def read_uid(in_name='data/music_users_count.txt', larger=100):
    uid_list = []
    for line in open(in_name):
        uid, count = line.strip().split(',')
        if int(count) > 100:
            uid_list.append(uid)
    return uid_list


if __name__ == '__main__':

    dir_name = '/home/kayzhou/exp/come_on_data/get_weibo_users/data/weibo_0320'
    uids_1 = set([line.strip() for line in open('uid/uid_-1.txt')])
    uids_2 = set([line.strip() for line in open('uid/uid_+1.txt')])
    uids = uids_1 | uids_2

    # 遍历微博数
#    i = 0
#    out_file = open('data/music_users_count.txt', 'w')
#    for in_name in os.listdir(dir_name):
#        i += 1
#        if i % 100 == 0: print(i)
#        cnt = how_many_weibo(dir_name + "/" + in_name)
#        out_file.write(in_name + ',' + str(cnt) + '\n')


    # 提取训练数据特征
#    out_file = open('data/features_20161008.txt', 'w')
#    uid_list = read_uid()
#    for in_name in uid_list:
#        # 考虑是否已经判断过微博个数
#        # if how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
#        #     continue
#
#        print(in_name)
#        try:
#            X = get_train_features(dir_name + "/" + in_name)
#            if X:
#                out_file.write(in_name + "," + ",".join([str(x) for x in X]) + "\n")
#        except:
#            traceback.print_exc(file=sys.stderr)


    # 提取需要分析的文本特征
    # out_file = open('large_425_shopping.txt', 'w')
    # for in_name in os.listdir(dir_name):
    #     if how_many_weibo(dir_name + "/" + in_name) < 100: # 爬取到的微博数小于100
    #         continue
    #     print(in_name)
    #     X = get_behavior_features(dir_name + "/" + in_name)
    #     out_file.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")

    # 提取关键词, 地理位置, url, 来源
    # out_keyword = open('large_510_keyword.txt', 'w', encoding='utf8')
    # out_geo = open('large_517_geo.txt', 'w', encoding='utf8')
    # out_url = open('large_510_url.txt', 'w', encoding='utf8')
    # out_source = open('large_510_source.txt', 'w', encoding='utf8')
    # for in_name in os.listdir(dir_name):
    #     if how_many_weibo(dir_name + "/" + in_name) < 1: # 爬取到的微博数小于100
    #         continue
    #     print(in_name)
    #     X = get_keyword_features(dir_name + "/" + in_name)
    #     print(X)
    #     out_keyword.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")
    #     X = get_geo_features(dir_name + "/" + in_name)
    #     print(X)
    #     out_geo.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")
    #     X = get_url_features(dir_name + "/" + in_name)
    #     print(X)
    #     out_url.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")
    #     X = get_source_features(dir_name + "/" + in_name)
    #     print(X)
    #     out_source.write(in_name + " " + " ".join([str(x) for x in X]) + "\n")

    # 提取徽章信息
    '''
    out_file = open('data/large_0224_badge.txt', 'w')
    for in_name in os.listdir(dir_name):
        if in_name not in uids:
            continue
        print(in_name)
        try:
            if in_name in uids_1:
                clas = '-1'
            elif in_name in uids_2:
                clas = '1'
            X = get_badge_features(dir_name + "/" + in_name)
            out_file.write(in_name + " " + clas + " " + " ".join([str(x) for x in X]) + "\n")
        except Exception as e:
            print('error =>', e, in_name)
    '''

    # 提取情绪信息
    out_file = open('data/large_0224_mood.txt', 'w')
    for in_name in os.listdir(dir_name):
        if in_name not in uids:
            continue
        print(in_name)
        try:
            if in_name in uids_1:
                clas = '-1'
            elif in_name in uids_2:
                clas = '1'
            X = get_mood_features(dir_name + "/" + in_name)
            out_file.write(in_name + " " + clas + " " + " ".join([str(x) for x in X]) + "\n")
        except Exception as e:
            print('error =>', e, in_name)

    # 时间特征
    out_file = open('data/large_0224_time.txt', 'w')
    for in_name in os.listdir(dir_name):
        if in_name not in uids:
            continue
        print(in_name)
        try:
            if in_name in uids_1:
                clas = '-1'
            elif in_name in uids_2:
                clas = '1'
            X = exact_tweet_datetime(dir_name + "/" + in_name)
            out_file.write(in_name + " " + clas + " " + " ".join([str(x) for x in X]) + "\n")
        except Exception as e:
            print('error =>', e, in_name)
