# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd


def split_2class(feature_in_name, out_name_n1, out_name_1, tag_name):
    '''
    根据 tag_name 将特征文件分为两个文件 (某一性格维度上高或低)
    :param feature_in_name:
    :param out_name_n1:
    :param out_name_1:
    :param tag_name:
    :return:
    '''
    dict_tags = {}
    for line in open(tag_name):
        t = line.strip().split(" ")
        dict_tags[t[0]] = t[1]

    out_file_n1 = open(out_name_n1, 'w')
    out_file_1 = open(out_name_1, 'w')
    for line in open(feature_in_name).readlines():
        uid = line[:10]
        try:
            tag = dict_tags[uid]
        except KeyError:
            print("缺失:", uid)
            tag = 'error'

        # 此处需要看 'feature_in_name' 实际数据格式
        for_write = line
        # for_write = line[11:]
        if tag == '-1.0':
            out_file_n1.write(for_write)
        elif tag == '1.0':
            out_file_1.write(for_write)


def mean_var(in_name):
    data = pd.read_csv(in_name, sep=' ', header=None, index_col=0)
    r, c = data.shape
    print(data)
    for i in np.arange(1, c+1):
        print(data[i].describe())
        print('-----------------------')


if __name__ == '__main__':

    # split_2class('data/large_IGNORE_331.txt', 'data/split_class/large_IGNORE_331_0_n1.txt',
    #              'data/split_class/large_IGNORE_331_0_1.txt', 'data/tags/328_IGNORE_sides_0.txt')
    # split_2class('data/large_IGNORE_331.txt', 'data/split_class/large_IGNORE_331_1_n1.txt',
    #              'data/split_class/large_IGNORE_331_1_1.txt', 'data/tags/large_tag_311_1.txt')
    # split_2class('data/large_IGNORE_331.txt', 'data/split_class/large_IGNORE_331_2_n1.txt',
    #              'data/split_class/large_IGNORE_331_2_1.txt', 'data/tags/328_IGNORE_sides_2.txt')
    # split_2class('data/large_IGNORE_331.txt', 'data/split_class/large_IGNORE_331_3_n1.txt',
    #              'data/split_class/large_IGNORE_331_3_1.txt', 'data/tags/328_IGNORE_sides_3.txt')
    # split_2class('data/large_IGNORE_331.txt', 'data/split_class/large_IGNORE_331_4_n1.txt',
    #              'data/split_class/large_IGNORE_331_4_1.txt', 'data/tags/large_tag_311_4.txt')

    split_2class('data/features/large_401_badge.txt', 'data/split_class/large_IGNORE_404_badge_-1.txt',
                 'data/split_class/large_IGNORE_404_badge_+1.txt', 'data/tags/large_404_IGNORE_1_NOR.txt')
    # split_2class('data/features_328_shopping.txt', 'data/split_class/large_IGNORE_331_4_shopping_n1.txt',
    #              'data/split_class/large_IGNORE_331_4_shopping_1.txt', 'data/tags/328_IGNORE_sides_4.txt')


    # mean_var('data/split_class/large_IGNORE_331_4_n1.txt')
    # mean_var('data/split_class/large_IGNORE_331_4_1.txt')

    # mean_var('data/split_class/large_IGNORE_331_4_shopping_n1.txt')
    # mean_var('data/split_class/large_IGNORE_331_4_shopping_1.txt')







