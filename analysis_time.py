# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np
import pandas as pd
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def day_night(in_name, out_name):
    '''
    发微博的时间序列转化为4个时间段的统计
    :param in_name:
    :param out_name:
    :return:
    '''

    out_file = open(out_name, 'w')
    for line in open(in_name, encoding='utf8').readlines():
        # cnt = [0] * 4
        cnt = [0] * 24
        uid = line[:11]
        times = line[11: ].strip().split(' ')
        for t in times:
            hour = t.split(',')[1][: 2]
            # print(hour)
            # 分为四个时间段
            # cnt[int(int(hour) / 6)] += 1
            # 分为24小时
            cnt[int(hour)] += 1
        out_file.write(uid + ' '.join([str(c) for c in cnt]) + '\n')


def weeks(in_name, out_name):
    '''
    发微博的时间序列转化为7个时间段的统计
    :param in_name:
    :param out_name:
    :return:
    '''

    out_file = open(out_name, 'w')
    for line in open(in_name, encoding='utf8').readlines():
        # cnt = [0] * 4
        cnt = [0] * 7
        uid = line[:11]
        times = line[11: ].strip().split(' ')
        for t in times:
            # print(t)
            wd = parse(t).weekday()
            # print(hour)
            # 分为四个时间段
            # cnt[int(int(hour) / 6)] += 1
            # 分为24小时
            cnt[int(wd)] += 1
        out_file.write(uid + ' '.join([str(c) for c in cnt]) + '\n')


def day_night_individual(in_name, out_name):
    '''
    小平均, 每个用户在不同时刻发微博的比例
    发微博的时间序列转化为4个时间段的统计
    :param in_name:
    :param out_name:
    :return:
    '''
    count = 0
    out_file = open(out_name, 'w')
    for line in open(in_name, encoding='utf8').readlines():
        count += 1; print('count:', count)
        cnt = [0] * 24
        uid = line[:11]
        times = line[11: ].strip().split(' ')
        for t in times:

            # 区别 work 和 rest
            wd = datetime.datetime.strptime(t, '%Y-%m-%d,%H:%M:%S').weekday()
            # if wd in [0, 1, 2, 3, 4]:
            if wd in [5, 6]:
                continue

            hour = t.split(',')[1][: 2]
            # print(hour)
            # 分为四个时间段
            # cnt[int(int(hour) / 6)] += 1
            # 分为24小时
            cnt[int(hour)] += 1

        if sum(cnt) > 0:
            out_file.write(uid + ' '.join([str(c / sum(cnt)) for c in cnt]) + '\n')


def day_night_mean(in_name):
    data = pd.read_csv(in_name, header=None, sep=' ')
    # print(data)
    cnt = []
    # 大平均
    for i in np.arange(data.shape[1]):
        if i == 0: continue
        cnt.append(sum(data[i]))
    # print(cnt)

    # 4个时间段
    # print(cnt[0] / sum(cnt), cnt[1] / sum(cnt), cnt[2] / sum(cnt), cnt[3] / sum(cnt))
    # 早晚
    # print(cnt[0] / sum(cnt) + cnt[3] / sum(cnt), cnt[1] / sum(cnt) + cnt[2] / sum(cnt))
    y = []
    for i in np.arange(len(cnt)):
        # print(i, cnt[i] / sum(cnt))
        print(cnt[i] / sum(cnt))
        y.append(cnt[i] / sum(cnt))
    x = np.arange(len(cnt))
    plt.plot(x, y)
    plt.show()
    print('----------')


def day_night_mean_compare(in_name, in_name2):
    data = pd.read_csv(in_name, header=None, sep=' ')
    # print(data)
    cnt = []
    # 大平均
    for i in np.arange(data.shape[1]):
        if i == 0: continue
        cnt.append(sum(data[i]))
    # print(cnt)

    # 4个时间段
    # print(cnt[0] / sum(cnt), cnt[1] / sum(cnt), cnt[2] / sum(cnt), cnt[3] / sum(cnt))
    # 早晚
    # print(cnt[0] / sum(cnt) + cnt[3] / sum(cnt), cnt[1] / sum(cnt) + cnt[2] / sum(cnt))
    y = []
    for i in np.arange(len(cnt)):
        # print(i, cnt[i] / sum(cnt))
        # print(cnt[i] / sum(cnt))
        y.append(cnt[i] / sum(cnt))
    print(sum(y[1: 8]))
    print(sum(y[8: 19]))
    print(sum(y[19: ] + y[0]))
    print('----------')


    data = pd.read_csv(in_name2, header=None, sep=' ')
    # print(data)
    cnt = []
    # 大平均
    for i in np.arange(data.shape[1]):
        if i == 0: continue
        cnt.append(sum(data[i]))
    # print(cnt)

    # 4个时间段
    # print(cnt[0] / sum(cnt), cnt[1] / sum(cnt), cnt[2] / sum(cnt), cnt[3] / sum(cnt))
    # 早晚
    # print(cnt[0] / sum(cnt) + cnt[3] / sum(cnt), cnt[1] / sum(cnt) + cnt[2] / sum(cnt))
    y2 = []
    for i in np.arange(len(cnt)):
        # print(i, cnt[i] / sum(cnt))
        # print(cnt[i] / sum(cnt))
        y2.append(cnt[i] / sum(cnt))
    print(sum(y2[1: 8]))
    print(sum(y2[8: 19]))
    print(sum(y2[19: ] + y[0]))
    print('----------')

    # 画图
    x = np.arange(len(cnt))
    opacity = 0.9
    # --- for *.eps --- #
    fig = plt.figure(figsize=(8, 6))

    fig.set_rasterized(True)
    print(x)
    print(y)
    print(y2)
    plt.plot(x, y, '-o', alpha=opacity, color=(1, 0.4, 0.2), label='Extroverts')
    plt.plot(x, y2, '-o', alpha=opacity, color=(0.2, 0.0, 0.9), label='Introverts')
    plt.xlim(0, 23)
    plt.xticks([0, 4, 8, 12,16, 20, 23], fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc=4)
    plt.xlabel("o'clock", fontsize=18)
    plt.ylabel("Proportion", fontsize=18)
    # plt.savefig('figure/time_series.eps', dpi=300)
    # plt.show()


def time_interval(in_name, out_name):
    out_file = open(out_name, 'w')
    for line in open(in_name, encoding='utf8').readlines():
        interval = []
        uid = line[:11]
        times = line[11: ].strip().split(' ')
        dts = []
        for t in times:
            dts.append(datetime.datetime.strptime(t, '%Y-%m-%d,%H:%M:%S'))
        for i in np.arange(1, len(dts)):
            interval.append((dts[i] - dts[i-1]).days * 86400 + (dts[i] - dts[i-1]).seconds)
            # print((dts[i] - dts[i-1]).days * 86400 + (dts[i] - dts[i-1]).seconds)
        out_file.write(uid + ' '.join([str(inter) for inter in interval]) + '\n')


def time_interval_mean_std(in_name):

    # 平均每个人
    total_interval_mean = 0
    total_interval_std = 0
    lines = 0
    for line in open(in_name).readlines():
        lines += 1
        interval = np.array([float(inter) for inter in line[11: ].strip().split(' ') if float(inter) < 86400])
        # interval = np.array([float(inter) for inter in line[11: ].strip().split(' ')])
        total_interval_mean += interval.mean()
        total_interval_std += interval.std()
    print('-- 期望 --')
    print(total_interval_mean / lines / 3600)
    print('-- 标准差 --')
    print(total_interval_std / lines / 3600)

    # 大平均
    # interval = []
    # for line in open(in_name).readlines():
    #
    #     interval += [
    #                     # float(inter) for inter in line[11: ].strip().split(' ') if float(inter) < 86400 # 只考虑一天之内的时间间隔
    #                     float(inter) for inter in line[11: ].strip().split(' ')
    #                 ]
    # print('-- 期望 --')
    # print(np.array(interval).mean() / 3600)
    # print('-- 标准差 --')
    # print(np.array(interval).std() / 3600)


def time_interval_var(in_name):

    total_interval_var = 0
    lines = 0
    for line in open(in_name).readlines():
        lines += 1
        interval = np.array([float(inter) for inter in line[11: ].strip().split(' ') if float(inter) < 86400])
        interval_mean = interval.var()
        total_interval_var += interval_mean
    print(total_interval_var / lines)


if __name__ == '__main__':

    # 统计发微博时段
    # day_night('data/features/large_510_time.txt', 'data/features/large_510_day_night.txt')
    # day_night('data/features/large_510_time.txt', 'data/features/large_510_24.txt')
    # day_night_individual('data/features/large_510_time.txt', 'data/features/large_510_24_individual.txt')
    # day_night_individual('data/features/large_510_time.txt', 'data/features/large_510_24_individual.txt')

    # 区分工作日和休息日
    # day_night_individual('data/features/large_510_time.txt', 'data/features/large_510_24_work.txt')
    # day_night_individual('data/features/large_510_time.txt', 'data/features/large_510_24_rest.txt')
    # weeks('data/features/large_510_time.txt', 'data/features/large_510_week.txt')

    # 累加
    # day_night_mean('data/split_class/large_510_day_night_-1.txt')
    # day_night_mean('data/split_class/large_510_day_night_+1.txt')
    # day_night_mean('data/features/large_510_24_-1.txt')
    # day_night_mean('data/features/large_510_24_+1.txt')
    # day_night_mean_compare('data/features/large_510_24_individual_+1.txt', 'data/features/large_510_24_individual_-1.txt')
    # day_night_mean_compare('data/features/large_510_24_work_+1.txt', 'data/features/large_510_24_work_-1.txt')
    # day_night_mean_compare('data/features/large_510_24_rest_+1.txt', 'data/features/large_510_24_rest_-1.txt')

    # time_interval('data/features/large_510_time.txt', 'data/features/large_510_interval.txt')

    # 平均期望
    time_interval_mean_std('data/split_class/large_510_interval_+1.txt')
    time_interval_mean_std('data/split_class/large_510_interval_-1.txt')
    # 平均方差
    # time_interval_var('data/split_class/large_510_interval_-1.txt')
    # time_interval_var('data/split_class/large_510_interval_+1.txt')

    pass