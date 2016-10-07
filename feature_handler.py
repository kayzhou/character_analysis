# -*- coding: utf-8 -*-
__author__ = 'Kay'

import json
import os
import numpy as np
import datetime
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from math import log
import re
import pandas as pd


'''
提取微博用户特征, 用于用户性格分析
'''

def duplicate_removal(in_name='data/train_data.txt'):
    user_id = set()
    out_file = open('data/user_id.txt')
    for line in open(in_name):
        if line.startswith('#'): continue
        user_id.add(line.split('\t')[0])
    out_file.write()


def get_user_id(in_dir='data/users_20160302', out_name='data/user_id.txt'):
    out_file = open(out_name, 'w')
    for file_name in os.listdir(in_dir):
        print(file_name)
        out_file.write(file_name + '\n')
    out_file.close()


def get_psy_test():
    '''
    抽取有效的心理测试数据
    :return:
    '''
    user_id = read_user_id()
    i = 0

    out_file = open('data/psy_test.txt', 'w')

    for line in open('data/train_data.txt'):
        if line.startswith('#'): continue
        uid = line.strip().split('\t')[0]
        if uid == user_id[i]:
            out_file.write(line)
            i += 1
    out_file.close()


def read_user_id(in_name='data/user_id.txt'):
    '''
    获取实验所用的user_id_set
    :param in_name:
    :return:
    '''
    user_id = list()
    for line in open(in_name):
        if line.startswith('#'): continue
        line = line.strip()
        line.split('\t')
        user_id.append(line.strip())
    return user_id


def read_psy_test_data(in_name='data/psy_test.txt'):
    count = 0
    train_data = {}
    for line in open(in_name):
        line = line.strip()
        if count == 0: count += 1; continue
        items = line.split('\t')
        train_data[items[0]] = {'gender': items[1], 'age': items[2], 'neu': items[3],
                                'ext': items[4], 'ope': items[5], 'agr': items[6], 'con': items[7]}
    return train_data


def read_user_data(in_name, out_name):
    '''
    抽取user字段,写入文件
    :param in_name:
    :param out_name:
    :return:
    '''
    in_file = open(in_name)
    out_file = open(out_name, 'a')
    file_set = set()
    for line in in_file:
        if in_file in file_set:
            continue
        else:
            file_set.add(in_file)
        line = json.loads(line.strip())
        print(line['user'])
        out_file.write(json.dumps(line['user']) + '\n')


def read_keyword_set():
    return list([line.strip() for line in open('data/word_list/keyword.txt', encoding='utf8')])


# def read_text_data(in_name, out_name):
#     '''
#     抽取text字段,写入文件
#     :param in_name:
#     :param out_name:
#     :return:
#     '''
#     keyword_set = read_keyword_set()
#     in_file = open(in_name)
#     out_file = open(out_name, 'a', encoding='utf8')
#     for line in in_file:
#         line = json.loads(line.strip())
#         text = line['text'].replace('\n', '')
#
#         # 判断是否有关键词
#         for keyword in keyword_set:
#             if keyword in text:
#                 # print(text)
#                 out_file.write(seg_word(filter(text)) + '\n')
#                 # out_file.write(filter(text) + '\n')
#                 # out_file.write(text + '\n')
#                 break


# def save_related_words(in_name, out_name_dir):
#     '''
#     保存相关词, 词的共现
#     :param in_name:
#     :param out_name:
#     :return:
#     '''
#     def read_keyword_set():
#         return set([line.strip() for line in open('data/keyword.txt', encoding='utf8')])
#
#     print(in_name)
#     keyword_set = read_keyword_set()
#     # print(len(keyword_set))
#     in_file = open(in_name)
#     for line in in_file:
#         line = json.loads(line.strip())
#         text = line['text'].replace('\n', '')
#
#         # 判断是否有关键词
#         for keyword in keyword_set:
#             if keyword in text:
#                 # print(text)
#                 open(out_name_dir + '/' + keyword, 'a', encoding='utf8').write(seg_word(filter(text)) + '\n')
#                 # out_file.write(filter(text) + '\n')
#                 # out_file.write(text + '\n')
#                 break


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


def line_static_features(line, last_dt):
    def b2i(bo):
            '''
            boolean类型转成可识别的int
            :param bo:
            :return:
            '''
            return str(int(bo))

    x = []
    raw_data = json.loads(line.strip())['user']

    # 性别
    if raw_data['gender'] == 'm':
        x.append('1')
    else:
        x.append('0')

    # 从注册到最后一条微博的天数: 注册天数
    reg_dt = str2datetime(raw_data['created_at'])
    res = (last_dt - reg_dt).days + 1
    # print("最后一条微博发布时间:", last_dt)
    # print("注册时间:", reg_dt)
    if res < 0: res = 1 # 异常情况
    x.append(log(res + 1))
    # log (微博数 + 1)
    x.append(log(float(raw_data['statuses_count']) + 1))
    # log (微博数 / 注册天数)
    x.append(float(raw_data['statuses_count']) / res)
    # log (关注数 + 1)
    x.append(log(float(raw_data['friends_count']) + 1))
    # log (粉丝数 + 1)
    x.append(log(float(raw_data['followers_count']) + 1))
    # 微博数 / (关注数 + 1)
    x.append(float(raw_data['statuses_count']) / (float(raw_data['friends_count']) + 1))
    # 微博数 / (粉丝数 + 1)
    x.append(float(raw_data['followers_count']) / (float(raw_data['friends_count']) + 1))

    # 是否为认证用户
    x.append(b2i(raw_data['verified']))
    # 是否允许评论
    x.append(b2i(raw_data['allow_all_comment']))
    # 是否允许发送私信
    x.append(b2i(raw_data['allow_all_act_msg']))
    # 是否开启地理位置
    x.append(b2i(raw_data['geo_enabled']))
    # 个人描述的长度
    x.append(len(raw_data['description']))
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
        x.append(b2i(raw_data["taobao"]))
    else:
        x.append("0")
    # 红包 2015
    if "hongbao_2015" in raw_data:
        x.append(b2i(raw_data["taobao"]))
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


def last_weibo(in_name, order):
    data = load_user_data(in_name)

    if (data[0]['created_at'] > data[-1]['created_at']) and order:
        raise KeyboardInterrupt

    if order == True:
        return data[-1]
    else:
        return data[0]


def extract_static_features(in_dir='data/users_20160302', out_name='data/features/static_feature.txt', order=False):
    '''
    抽出用户静态特征
    :param in_name:
    :param out_name:
    :return:
    '''
    out_file = open(out_name, 'w')
    for file_name in os.listdir(in_dir):
        if os.path.isfile(os.path.join(in_dir, file_name)) and file_name in psy_test_data:
            in_name = os.path.join(in_dir, file_name)
            print(file_name)
            last_w = last_weibo(in_name, order)
            line = json.dumps(last_w)
            X = line_static_features(line, str2datetime(last_w['created_at']))
            out_file.write(file_name + ' ' + ' '.join([str(x) for x in X]) + '\n')


def how_many_weibo(in_name):
    return len(open(in_name, encoding='utf8').readlines())


def file_dynamic_features(in_name):
    # 获取微博发送的日期时间
    dts = exact_tweet_datetime(in_name)
    print(in_name, len(dts))
    # 多少周发送微博, 多少天发送微博 (该周或天发送一条即可)
    cnt_weeks, cnt_days = exact_how_many_weeks_days(dts)
    # 提取时间序列的特征
    X = exact_series_feature(exact_day_series(dts), cnt_days) \
        + exact_series_feature(exact_week_series(dts), cnt_weeks)
    # 为了过滤!
    # return X
    # 获取转发的日期时间
    retweet_dts = exact_tweet_datetime(in_name, action='retweet')
    X += (exact_series_feature(exact_day_series(retweet_dts), cnt_days) \
        + exact_series_feature(exact_week_series(retweet_dts), cnt_weeks))

    # 获取@的日期时间
    at_dts = exact_tweet_datetime(in_name, action='at')
    X += (exact_series_feature(exact_day_series(at_dts), cnt_days) \
        + exact_series_feature(exact_week_series(at_dts), cnt_weeks))

    # 转发比例
    X.append(len(retweet_dts) / (len(dts) + 1))
    # @比例
    X.append(len(at_dts) / (len(dts) + 1))
    # 微博平均长度
    X.append(extract_tweet_ave_length(in_name))
    return X


def extract_dynamic_features(in_dir='data/users_20160302', out_name='data/features/dynamic_feature.txt'):
    '''
    提取动态特征
    :param in_dir:
    :param out_name:
    :return:
    '''
    out_file = open(out_name, 'w')
    for file_name in os.listdir(in_dir):
        if os.path.isfile(os.path.join(in_dir, file_name)) and file_name in psy_test_data:
            X = file_dynamic_features(os.path.join(in_dir, file_name))
            out_file.write(file_name + ' ' + ' '.join([str(x) for x in X]) + '\n')

    out_file.close()


def classify_targets(out_name):
    '''
    心理测试的结果是连续的, 不易分析, 现在将问题转换成分类问题
    :param out_name:
    :return:
    '''
    user_id = read_user_id()
    psy_test_data = read_psy_test_data()
    out_file = open(out_name, 'w')
    for uid in user_id:
        print(psy_test_data[uid])
        x = []
        x.append('1' if int(psy_test_data[uid]['neu']) > 36 else '0')
        x.append('1' if int(psy_test_data[uid]['ext']) > 39 else '0')
        x.append('1' if int(psy_test_data[uid]['ope']) > 41 else '0')
        x.append('1' if int(psy_test_data[uid]['agr']) > 42 else '0')
        x.append('1' if int(psy_test_data[uid]['con']) > 41 else '0')

        out_file.write(' '.join([str(x) for x in x]) + '\n')


def classify_target_3(out_name):
    '''
    心理测试的结果是连续的, 不易分析, 现在将问题转换成分类问题
    :param out_name:
    :return:
    '''
    neu_mean = 36.02
    neu_std = 9.718 / 2
    ext_mean = 39.03
    ext_std = 7.549 / 2
    ope_mean = 40.96
    ope_std = 5.308 / 2
    agr_mean = 41.66
    agr_std = 5.443 / 2
    con_mean = 40.78
    con_std = 6.553 / 2

    user_id = read_user_id()
    psy_test_data = read_psy_test_data()
    out_file = open(out_name, 'w')
    for uid in user_id:
        print(psy_test_data[uid])
        x = []
        if float(psy_test_data[uid]['neu']) <= neu_mean - neu_std:
            x.append('-1')
        elif neu_mean - neu_std < float(psy_test_data[uid]['neu']) <= neu_mean + neu_std:
            x.append('0')
        else:
            x.append('1')

        if float(psy_test_data[uid]['ext']) <= ext_mean - ext_std:
            x.append('-1')
        elif ext_mean - ext_std < float(psy_test_data[uid]['ext']) <= ext_mean + ext_std:
            x.append('0')
        else:
            x.append('1')

        if float(psy_test_data[uid]['ope']) <= ope_mean - ope_std:
            x.append('-1')
        elif ope_mean - ope_std < float(psy_test_data[uid]['ope']) <= ope_mean + ope_std:
            x.append('0')
        else:
            x.append('1')

        if float(psy_test_data[uid]['agr']) <= agr_mean - agr_std:
            x.append('-1')
        elif agr_mean - agr_std < float(psy_test_data[uid]['agr']) <= agr_mean + agr_std:
            x.append('0')
        else:
            x.append('1')

        if float(psy_test_data[uid]['con']) <= con_mean - con_std:
            x.append('-1')
        elif con_mean - con_std < float(psy_test_data[uid]['con']) <= con_mean + con_std:
            x.append('0')
        else:
            x.append('1')


        out_file.write(' '.join([str(x) for x in x]) + '\n')


def load_classification(in_name='data/classify_train_data.txt'):
    raw_data = np.loadtxt(in_name, dtype=int)
    print(raw_data.take(0, axis=1))


def str2datetime(s):
    '''
    Example: Sat Jun 12 23:40:00 +0800 2010 -> datetime.datetime(2010, 6, 12, 23, 40)
    :param s:
    :return:
    '''
    if len(s) == 19:
        return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.strptime(s[:-11] + s[-5:], '%c')


def exact_tweet_datetime(in_name, action='tweet', output_type='dt'):
    '''
    提取微博发布日期
    :param in_name:
    :param out_name:
    :return:
    '''
    user_data = load_user_data(in_name)
    if action == 'tweet':
        if output_type == 'dt':
            return [str2datetime(user['created_at']) for user in user_data]
        elif output_type == 'str':
            return [str2datetime(user['created_at']).strftime('%Y-%m-%d,%H:%M:%S') for user in user_data]
    elif action == 'retweet':
        return [str2datetime(user['created_at']) for user in user_data if "retweeted_status" in user]
    elif action == "at":
        return [str2datetime(user['created_at']) for user in user_data
                if "@" in user["text"][:user["text"].find("//")] or ("retweeted_status" not in user and "@" in user["text"])]


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

    for user in user_data:
        tweet_length.append(len(filter(user['text'])))

    tweet_length = np.array(tweet_length)
    # print(tweet_length.mean())
    # print(tweet_length.max())
    # print(tweet_length.argmax())
    # print(tweet_length.min())
    # print(tweet_length.__len__())

    return tweet_length.mean()


def exact_how_many_weeks_days(dts):
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


def exact_week_series(dts):
    '''
    动态特征, 统计每周时间序列
    :param dts:
    :return:
    '''
    week_series = np.array([0] * 7)
    for dt in dts:
        week_series[dt.weekday()] += 1
    return week_series


def exact_day_series(dts):
    '''
    动态特征, 统计每天时间序列
    :param dts:
    :return:
    '''
    day_series = np.array([0] * 24)
    for dt in dts:
        day_series[dt.hour] += 1
    return day_series


def exact_series_feature(se, cnt):
    '''
    动态特征, 最大值所在, 最大值, 最小值所在, 最小值, 方差
    :param se:
    :return:
    '''
    # 均值, 最大所在, 最大平均, 最小所在, ¡¡¡¡¡平均的方差
    return [se.sum() / cnt, se.argmax(), se.max() / cnt, se.argmin(), (se / cnt).var()]


def exact_user_text(in_name):
    user_data = load_user_data(in_name)
    return [user['text'] for user in user_data]


def get_ignore_index():
    data = pd.read_csv('data/data_amount.txt', delimiter=" ", header=None)
    new_data = data[data[1]<100].any(1).index
    return new_data


def union_feature(in_name_feature, in_name_quality, out_name, target_index, ignore=True):
    '''
    将特征和目标合并
    :param in_name_feature: 特征文件
    :param in_name_quality: 目标文件
    :param out_name: 输出文件
    :return:
    '''
    feature_data = np.loadtxt(in_name_feature, dtype=float)
    quality_data = np.loadtxt(in_name_quality, dtype=int)
    out_file = open(out_name, 'w')
    if ignore:
        ignores = get_ignore_index()
    else:
        ignores = []

    for i in range(len(feature_data)):

        if ignore and i in ignores:
            continue

        y = quality_data[i][target_index]
        x = feature_data[i][1:] # 不考虑uid

        # out_file.write(str(y) + ' ' + ' '.join([str(i + 1) + ':' + str(xi) for i, xi in enumerate(x)]) + '\n')
        out_file.write(str(y) + ' ' + ' '.join([str(xi) for xi in x]) + '\n')
        # out_file.write(' '.join([str(yi) for yi in y]) + ' ' + ' '.join([str(xi) for xi in x]) + '\n')
    # dump_svmlight_file(x, y, out_name)


def union_feature_tfidf(in_dir_feature, in_name_quality, out_name, target_index):
    '''
    将 tfidf值 和 目标 合并
    :param in_name_feature:
    :param in_name_quality:
    :param out_name:
    :return:
    '''
    quality_data = np.loadtxt(in_name_quality, dtype=int)
    out_file = open(out_name, 'w', encoding='utf8')

    i = 0
    for file_name in os.listdir(in_dir_feature):
        if not file_name.endswith('.txt'): continue
        # 选择目标 0-4
        print(file_name)
        y = quality_data[i][target_index]
        x = np.loadtxt(os.path.join(in_dir_feature, file_name)).tolist()
        # print(y, x)
        out_file.write(str(y) + '\t' + '\t'.join([str(i + 1) + ':' + str(xi) for i, xi in enumerate(x) if xi != 0]) + '\n')
        i += 1


def union_feature_sides(in_name_feature, in_name_quality, out_name, target_index, ignore=True):
    '''
    将特征和目标合并
    :param in_name_feature: 特征文件
    :param in_name_quality: 目标文件
    :param out_name: 输出文件
    :return:
    '''
    feature_data = np.loadtxt(in_name_feature, dtype=float)
    quality_data = np.loadtxt(in_name_quality, dtype=int)
    out_file = open(out_name, 'w')

    if ignore:
        ignores = get_ignore_index()

    for i in range(len(feature_data)):

        if ignore and i in ignores:
            continue
        # y = quality_data[i][goal]
        y = quality_data[i][target_index]
        if y == 0: continue
        x = feature_data[i][1:] # 不考虑uid

        # out_file.write(str(y) + ' ' + ' '.join([str(i + 1) + ':' + str(xi) for i, xi in enumerate(x)]) + '\n')
        out_file.write(str(y) + ' ' + ' '.join([str(xi) for xi in x]) + '\n')
        # out_file.write(' '.join([str(yi) for yi in y]) + ' ' + ' '.join([str(xi) for xi in x]) + '\n')



if __name__ == '__main__':
    user_id = read_user_id()
    psy_test_data = read_psy_test_data()

    # X, y = load_svmlight_file('/Users/Kay/Project/EXP/character_analysis/data/SVM/20160221_tfidf_0.txt')
    # print(X)
    # print(y)

    # 离散目标
    # classify_goals('data/classify_train_data.txt')
    # classify_goals('data/regress_train_data.txt')
    # classify_target_3('data/classify3_train_data.txt')


    # 写入用户静态数据
    # for uid in user_id:
    #     read_user_data(os.path.join('data/users_20160302', uid), 'data/static_user_data.txt')

    # 写入用户微博文本数据
    # for uid in user_id:
    #     read_text_data(os.path.join('data/users_20160302', uid), 'data/text_data.txt')
    # for uid in user_id:
    #     save_related_words(os.path.join('data/users_20160302', uid), 'data/related_weibo')

    # 静态特征
    # extract_static_features()

    # 动态特征
    # extract_dynamic_features()

    #  整合三种特征
    # sf = open('data/features/static_feature.txt').readlines()
    # df = open('data/features/dynamic_feature.txt').readlines()
    # tf = open('data/features/word_appear_scale.txt').readlines()
    # f = open('data/features/328_features.txt', 'w')
    # for i in range(len(sf)):
    #     f.write(sf[i].strip() + ' ' + df[i][11:].strip() + ' ' + tf[i])
    # f.close()

    # 将特征和目标结合
    union_feature('data/features/train_IGNORE_404.txt', 'data/regress_train_data.txt', 'data/for_analysis/404_IGNORE_NOR_0.txt', 0)
    union_feature('data/features/train_IGNORE_404.txt', 'data/regress_train_data.txt', 'data/for_analysis/404_IGNORE_NOR_1.txt', 1)
    union_feature('data/features/train_IGNORE_404.txt', 'data/regress_train_data.txt', 'data/for_analysis/404_IGNORE_NOR_2.txt', 2)
    union_feature('data/features/train_IGNORE_404.txt', 'data/regress_train_data.txt', 'data/for_analysis/404_IGNORE_NOR_3.txt', 3)
    union_feature('data/features/train_IGNORE_404.txt', 'data/regress_train_data.txt', 'data/for_analysis/404_IGNORE_NOR_4.txt', 4)

    # union_feature('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_0.txt', 0)
    # union_feature('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_1.txt', 1)
    # union_feature('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_2.txt', 2)
    # union_feature('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_3.txt', 3)
    # union_feature('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_4.txt', 4)

    # union_feature_sides('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_sides_0.txt', 0)
    # union_feature_sides('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_sides_1.txt', 1)
    # union_feature_sides('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_sides_2.txt', 2)
    # union_feature_sides('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_sides_3.txt', 3)
    # union_feature_sides('data/features/train_IGNORE_404_NOR.txt', 'data/classify3_train_data.txt', 'data/SVM/404_NOR_features_sides_4.txt', 4)

    # 将特征和目标结合
    # union_feature_tfidf('data/tfidf_scale', 'data/classify3_train_data.txt', 'data/SVM/314_tfidf_scale_0.txt', 0)
    # union_feature_tfidf('data/tfidf_scale', 'data/classify3_train_data.txt', 'data/SVM/314_tfidf_scale_1.txt', 1)
    # union_feature_tfidf('data/tfidf_scale', 'data/classify3_train_data.txt', 'data/SVM/314_tfidf_scale_2.txt', 2)
    # union_feature_tfidf('data/tfidf_scale', 'data/classify3_train_data.txt', 'data/SVM/314_tfidf_scale_3.txt', 3)
    # union_feature_tfidf('data/tfidf_scale', 'data/classify3_train_data.txt', 'data/SVM/314_tfidf_scale_4.txt', 4)