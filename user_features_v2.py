# -*- coding: utf-8 -*-
__author__ = 'Kay'

import datetime
import json
import os
import re
from math import log
# import linecache

import arrow
import numpy as np
import pandas as pd


'''
提取微博用户特征
'''

def str2datetime(s):
    '''
    字符串转datetime类
    '''
    return arrow.get(s)


def load_user_data(in_name):
    '''
    载入用户全部数据
    :param in_name:
    :return:
    '''
    user_data = []
    for line in open(in_name):
        line = line.strip()
        user_data.append(json.loads(line))
    return user_data


def static_features(d):

    def b2i(bo):
        '''
        boolean类型转成可识别的int
        :param bo:
        :return:
        '''
        return str(int(bo))

    x = []

    last_dt = arrow.get(d['pDate'])
    d = d['from']

    # 性别
    if d['extend']['gender'] == 'm':
        x.append('1')
    else:
        x.append('0')

    # 从注册到最后一条微博的天数: 注册天数
    reg_dt = arrow.get(d['extend']['created_at'], 'ddd MMM DD HH:mm:ss Z YYYY')
    res = (last_dt - reg_dt).days + 1

    if res < 0:
        res = -1 # 异常情况

    # 注册天数
    x.append(log(res + 1))
    # log (微博数 + 1)
    x.append(log(float(d['extend']['statuses_count']) + 1))
    # log (微博数 / 注册天数)
    x.append(float(d['extend']['statuses_count']) / res)
    # log (关注数 + 1)
    x.append(log(float(d['extend']['friends_count']) + 1))
    # log (粉丝数 + 1)
    x.append(log(float(d['extend']['followers_count']) + 1))
    # 微博数 / (关注数 + 1)
    x.append(float(d['extend']['statuses_count']) / (float(d['extend']['friends_count']) + 1))
    # 微博数 / (粉丝数 + 1)
    x.append(float(d['extend']['followers_count']) / (float(d['extend']['friends_count']) + 1))

    # 是否为认证用户
    x.append(b2i(d['extend']['verified']))
    # 是否允许评论
    x.append(b2i(d['extend']['allow_all_comment']))
    # 是否允许发送私信
    x.append(b2i(d['extend']['allow_all_act_msg']))
    # 是否开启地理位置
    x.append(b2i(d['extend']['geo_enabled']))
    # 个人描述的长度
    x.append(len(d['description']))
    return x


def line_get_badge(line):
    '''
    获得徽章信息
    :param line:
    :return:
    '''
    def b2i(bo):
        '''
        boolean类型转成可识别的int
        :param bo:
        :return:
        '''
        return str(int(bo))

    x = []
    raw_data = json.loads(line.strip())["user"]["badge"]
    print(raw_data)
    # 绑定淘宝
    if "bind_taobao" in raw_data:
        x.append(b2i(raw_data["bind_taobao"]))
    else:
        x.append("0")
    # 双十一
    if "shuang11_2015" in raw_data:
        x.append(b2i(raw_data["shuang11_2015"]))
    else:
        x.append("0")
    # 淘宝
    if "taobao" in raw_data:
        x.append(b2i(raw_data["taobao"]))
    else:
        x.append("0")
    # 红包 2014
    if "hongbao_2014" in raw_data:
        x.append(b2i(raw_data["hongbao_2014"]))
    else:
        x.append("0")
    # 红包 2015
    if "hongbao_2015" in raw_data:
        x.append(b2i(raw_data["hongbao_2015"]))
    else:
        x.append("0")

    # x.append(b2i(raw_data["uc_domain"]))
    # x.append(b2i(raw_data["enterprise"]))
    # x.append(b2i(raw_data["suishoupai_2014"]))
    # x.append(b2i(raw_data["zongyiji"]))
    # x.append(b2i(raw_data["gongyi_level"]))
    # x.append(b2i(raw_data["dailv"]))
    # x.append(b2i(raw_data["gongyi"]))
    # x.append(b2i(raw_data["travel2013"]))
    # x.append(b2i(raw_data["anniversary"]))
    # x.append(b2i(raw_data["pzsd_2015"]))

    print(x)
    return x


def last_weibo(in_name):
    d = load_user_data(in_name)
    return d[0]




    # out_file.write(file_name + ' ' + ' '.join([str(x) for x in X]) + '\n')


def dynamic_features(user_data):
    '''
    动态特征
    '''
    # 获取微博发送的日期时间
    dts = extract_tweet_datetime(user_data)

    # 多少周发送微博, 多少天发送微博 (该周或天发送一条即可)
    cnt_weeks, cnt_days = extract_how_many_weeks_days(dts)

    # 提取时间序列的特征
    X = extract_series_feature(extract_day_series(dts), cnt_days) \
        + extract_series_feature(extract_week_series(dts), cnt_weeks)

    # 获取转发的日期时间
    retweet_dts = extract_tweet_datetime(user_data, action='retweet')

    X += (extract_series_feature(extract_day_series(retweet_dts), cnt_days) \
        + extract_series_feature(extract_week_series(retweet_dts), cnt_weeks))

    # 获取@的日期时间
    at_dts = extract_tweet_datetime(user_data, action='at')

    X += (extract_series_feature(extract_day_series(at_dts), cnt_days) \
        + extract_series_feature(extract_week_series(at_dts), cnt_weeks))

    # 转发比例
    X.append(len(retweet_dts) / (len(dts) + 1))
    # @比例
    X.append(len(at_dts) / (len(dts) + 1))
    # 微博平均长度
    # X.append(extract_tweet_ave_length(in_name))
    return X



def extract_tweet_datetime(user_data, action='tweet', output_type='dt'):
    '''
    提取微博发布日期
    '''

    if action == 'tweet':
        if output_type == 'dt':
            return [str2datetime(d['pDate']) for d in user_data]
        elif output_type == 'str':
            return [str2datetime(d['pDate']).format('%Y-%m-%d,%H:%M:%S') for d in user_data]

    elif action == 'retweet':
        return [str2datetime(d['pDate']) for d in user_data if "retweeted_status" in d]

    elif action == "at":
        return [str2datetime(d['pDate']) for d in user_data
                if "@" in d["content"][:d["content"].find("//")] or ("retweeted_status" not in d and "@" in d["content"])]


def extract_mood(in_name):
    user_data = load_user_data(in_name)
    mood = [0] * 6
    for data in user_data:
        tag = data['mood']
        if tag == 0:
            mood[0] += 1
        elif tag == 1:
            mood[1] += 1
        elif tag == 2:
            mood[2] += 1
        elif tag == 3:
            mood[3] += 1
        elif tag == 4:
            mood[4] += 1
        elif tag == -1:
            mood[5] += 1
    return mood


def extract_tweet_ave_length(in_name):
    '''
    获取微博的长度
    :param in_name:
    :return:
    '''
    user_data = load_user_data(in_name)
    tweet_length = []

    def remove_retweet(tweet):
        return tweet[:tweet.find("//")]

    def remove_at(tweet):
        del_at=r'@.*?:|@.*?\s|@.*?$'
        tweet=re.sub(del_at,'',tweet)
        return tweet

    def remove_share(tweet):
        tweet=re.sub(r'\(分享自.*?\)','',tweet)
        return re.sub(r'（分享自.*?）','',tweet)

    def remove_emoticon(tweet):
        del_emo=r'\[.*?\]'
        return re.sub(del_emo,'',tweet)

    def remove_url(tweet):
        url_pattern=r'http://t.cn/\w+'
        tweet=re.sub(url_pattern,'',tweet)
        return tweet

    def filter(tweet):
        return remove_url(remove_at(remove_share(remove_retweet(tweet))))

    for d in user_data:
        tweet_length.append(len(filter(d['text'])))

    return np.array(tweet_length).mean()
    # print(tweet_length.mean())
    # print(tweet_length.max())
    # print(tweet_length.argmax())
    # print(tweet_length.min())
    # print(tweet_length.__len__())


def extract_how_many_weeks_days(dts):
    '''
    这其中有几天, 有几周
    :param dts:
    :return:
    '''
    def toMonday(dt):
        # 转成周一
        ndt = dt - datetime.timedelta(days = dt.weekday())
        return datetime.datetime(ndt.year, ndt.month, ndt.day)

    weeks = set()
    days = set()
    for dt in dts:
        weeks.add(toMonday(dt))
        days.add(datetime.datetime(dt.year, dt.month, dt.day))

    return len(weeks), len(days)


def extract_week_series(dts):
    '''
    动态特征, 统计每周时间序列
    :param dts:
    :return:
    '''
    week_series = np.array([0] * 7)
    for dt in dts:
        week_series[dt.weekday()] += 1
    return week_series


def extract_day_series(dts):
    '''
    动态特征, 统计每天时间序列
    :param dts:
    :return:
    '''
    day_series = np.array([0] * 24)
    for dt in dts:
        day_series[dt.hour] += 1
    return day_series


def extract_series_feature(se, cnt):
    '''
    动态特征, 最大值所在, 最大值, 最小值所在, 方差
    :param se:
    :return:
    '''
    # 均值, 最大所在, 最大平均, 最小所在, 平均的方差
    return [se.sum() / cnt, se.argmax(), se.max() / cnt, se.argmin(), (se / cnt).var()]


def how_many_weibo(in_name):
    if not os.path.exists(in_name):
        return -1

    return len(open(in_name).readlines())


def file_features(in_name):
    '''
    抽出用户静态特征
    :param in_name:
    :param out_name:
    :return:
    '''
    user_data = load_user_data(in_name)
    last_w = user_data[0] # 第一条是最新发出的微博
    X = []
    X_static = static_features(last_w)
    X_dynamic = dynamic_features(user_data)
    X.extend(X_static)
    X.extend(X_dynamic)

    return X


def get_user_id(in_dir):
    out_file = open('uid-count.txt', 'w')
    for i, in_name in enumerate(os.listdir(in_dir)):
        if not i % 100:
            print(i)
        uid = os.path.splitext(in_name)[0]
        count = how_many_weibo(os.path.join(in_dir, in_name))
        # print(uid + ',' + str(count))
        out_file.write(uid + ',' + str(count) + '\n')
    out_file.close()


def load_user_id(in_name):
    d = pd.read_csv(in_name)
    return d[d['count'] >= 100]['uid']


def list_to_str(li):
    new_li = []
    for l in li:
        if isinstance(l, float):
            new_li.append(str(round(l, 3)))
        else:
            new_li.append(str(l))
    return ','.join(new_li)


if __name__ == '__main__':
    # get_user_id('data/user_data')
    uids = load_user_id('data/uid-count.txt')

    out_file = open('weibo-features-20170606.txt', 'w')
    for i, uid in enumerate(uids):
        if not i % 100:
            print(i)
        in_name = 'data/user_data/{}.txt'.format(uid)
        X = file_features(in_name)
        out_file.write(str(uid) + ',' + list_to_str(X) + '\n')
    out_file.close


