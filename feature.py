# -*- coding: utf-8 -*-
__author__ = 'Kay'

from feature_handler import *
from sklearn.feature_extraction.text import CountVectorizer
import json


def get_static_features(line):
    def b2i(bo):
        '''
        boolean类型转成可识别的int
        :param bo:
        :return:
        '''
        return str(int(bo))

    x = []
    raw_data = json.loads(line.strip())['user']

    if raw_data['gender'] == 'm':
        x.append('1')
    else:
        x.append('0')

    # 注册到2016年3月18日的天数
    res = (datetime.datetime(2016, 3, 18)-str2datetime(raw_data['created_at'])).days
    x.append(log(res))
    x.append(log(float(raw_data['statuses_count'])))
    x.append(float(raw_data['statuses_count']) / res)
    x.append(log(float(raw_data['friends_count']) + 1))
    x.append(log(float(raw_data['followers_count']) + 1))
    x.append(float(raw_data['statuses_count']) / (float(raw_data['friends_count']) + 1))
    x.append(float(raw_data['followers_count']) / (float(raw_data['friends_count']) + 1))

    x.append(b2i(raw_data['verified']))
    x.append(b2i(raw_data['allow_all_comment']))
    x.append(b2i(raw_data['allow_all_act_msg']))
    x.append(b2i(raw_data['geo_enabled']))

    x.append(len(raw_data['description']))
    return x


def str2datetime(s):
    try:
        re = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except:
        re = datetime.datetime.strptime(s[:-11] + s[-5:], '%c')
    return re


def get_dynamic_features(dts):
    cnt_weeks, cnt_days = exact_how_many_weeks_days(dts)
    x = exact_series_feature(exact_day_series(dts), cnt_weeks) \
        + exact_series_feature(exact_week_series(dts), cnt_days)
    print(type(x))
    return x


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
    all_text = [""]
    all_dts = []
    for line in open(in_name, encoding='utf8'):
        json_data = json.loads(line.strip())
        # print(json_data)
        all_text[0] += json_data["text"]
        all_dts.append(str2datetime(json_data["created_at"]))

    return get_static_features(one_line) + get_dynamic_features(all_dts) + get_text_features(all_text)


if __name__ == '__main__':
    # print(get_text_features('我是一个正直而且自信的人.'))
    print(get_features("/Users/Kay/Project/EXP/character_analysis/1080807495"))