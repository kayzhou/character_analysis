# -*- coding: utf-8 -*-
__author__ = 'Kay'


import numpy as np
import pandas as pd
import datetime


def day_night(in_name, out_name):
    '''
    发微博的时间序列转化为4个时间段的统计
    :param in_name:
    :param out_name:
    :return:
    '''
    out_file = open(out_name, 'w')
    for line in open(in_name, encoding='utf8').readlines():
        cnt = [0, 0, 0, 0]
        uid = line[:11]
        times = line[11: ].strip().split(' ')
        for t in times:
            hour = t.split(',')[1][: 2]
            # print(hour)
            cnt[int(int(hour) / 6)] += 1
            # print(int(int(hour) / 6))
        out_file.write(uid + ' '.join([str(c) for c in cnt]) + '\n')


def day_night_mean(in_name):
    data = pd.read_csv(in_name, header=None, sep=' ')
    # print(data)
    cnt = []
    cnt.append(sum(data[1]))
    cnt.append(sum(data[2]))
    cnt.append(sum(data[3]))
    cnt.append(sum(data[4]))
    # print(cnt)
    print(cnt[0] / sum(cnt), cnt[1] / sum(cnt), cnt[2] / sum(cnt), cnt[3] / sum(cnt))
    print(cnt[0] / sum(cnt) + cnt[3] / sum(cnt), cnt[1] / sum(cnt) + cnt[2] / sum(cnt))


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


def time_interval_mean(in_name):

    total_interval_mean = 0
    lines = 0
    for line in open(in_name).readlines():
        lines += 1
        interval = np.array([int(inter) for inter in line[11: ].strip().split(' ')])
        total_interval_mean += interval.mean()
    print(total_interval_mean / lines)



def time_interval_var(in_name):

    total_interval_var = 0
    lines = 0
    for line in open(in_name).readlines():
        lines += 1
        interval = np.array([int(inter) for inter in line[11: ].strip().split(' ')])
        interval_mean = interval.var()
        total_interval_var += interval_mean
    print(total_interval_var / lines)


if __name__ == '__main__':

    # 统计发微博时段
    # day_night('data/features/large_510_time.txt', 'data/features/large_510_day_night.txt')
    # 累加
    # day_night_mean('data/split_class/large_510_day_night_-1.txt')
    # day_night_mean('data/split_class/large_510_day_night_+1.txt')

    # time_interval('data/features/large_510_time.txt', 'data/features/large_510_interval.txt')
    # 平均期望
    # time_interval_mean('data/split_class/large_510_interval_-1.txt')
    # time_interval_mean('data/split_class/large_510_interval_+1.txt')
    # 平均方差
    time_interval_var('data/split_class/large_510_interval_-1.txt')
    time_interval_var('data/split_class/large_510_interval_+1.txt')