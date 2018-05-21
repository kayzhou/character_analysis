# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd

'''
对feature.py 产生的结果进行初步分析
'''

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
    for line in open(tag_name, encoding='utf8'):
        t = line.strip().split(" ")
        dict_tags[t[0]] = t[1]

    out_file_n1 = open(out_name_n1, 'w', encoding='utf8')
    out_file_1 = open(out_name_1, 'w', encoding='utf8')
    for line in open(feature_in_name, encoding='utf8').readlines():
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


def keyword_difference_set(in_name1, in_name2, out_name1, out_name2):
    '''
    计算关键词差集
    :param in_name1:
    :param in_name2:
    :return:
    '''
    def read_set(in_name):
        s = set()
        for line in open(in_name, encoding='utf8').readlines():
            s.add(line.split('\t')[0])
        return s


    def print_set(out_set, in_name, out_name):
        for line in open(in_name, encoding='utf8').readlines():
            if line.split('\t')[0] in out_set:
                open(out_name, 'a', encoding='utf8').write(line)


    s1 = read_set(in_name1)
    s2 = read_set(in_name2)
    out_set_1 = s1 - s2
    out_set_2 = s2 - s1
    print(len(out_set_1))
    print(len(out_set_2))
    print_set(out_set_1, in_name1, out_name1)
    print_set(out_set_2, in_name2, out_name2)


def Proportion(in_name, out_name):
    # data = pd.read_csv(in_name, header=None)
    data = pd.read_csv(in_name, header=None, sep='\t')
    s = sum(data[1])
    out = pd.DataFrame({'source': pd.Series(data[0]), 'count': pd.Series(data[1]), 'pro': pd.Series(data[1] / s)})
    out.to_csv(out_name, encoding='utf8')


if __name__ == '__main__':

    # 按性格分为两类, 打印到文件
    split_2class('data/features/large_510_24_work.txt',
                 'data/features/large_510_24_work_-1.txt',
                 'data/features/large_510_24_work_+1.txt',
                 'data/tags/large_404_IGNORE_1_NOR.txt')

    split_2class('data/features/large_510_24_rest.txt',
                 'data/features/large_510_24_rest_-1.txt',
                 'data/features/large_510_24_rest_+1.txt',
                 'data/tags/large_404_IGNORE_1_NOR.txt')

    # split_2class('data/features/large_517_geo.txt',
    #              'data/split_class/large_517_geo_-1.txt',
    #              'data/split_class/large_517_geo_+1.txt',
    #              'data/tags/large_404_IGNORE_1_NOR.txt')


    # mean_var('data/split_class/large_IGNORE_331_4_n1.txt')
    # mean_var('data/split_class/large_IGNORE_331_4_1.txt')

    # mean_var('data/split_class/large_IGNORE_331_4_shopping_n1.txt')
    # mean_var('data/split_class/large_IGNORE_331_4_shopping_1.txt')

    # keyword_difference_set('/Users/Kay/Project/EXP/character_analysis/keywords_ext_+1.txt',
    #                        '/Users/Kay/Project/EXP/character_analysis/keywords_ext_-1.txt',
    #                        '/Users/Kay/Project/EXP/character_analysis/keywords_ext_+1_diff.txt',
    #                        '/Users/Kay/Project/EXP/character_analysis/keywords_ext_-1_diff.txt')

    # 计算比例
    # Proportion('source_ext_-1.txt', 'source_pro_ext_-1.csv')
    # Proportion('source_ext_+1.txt', 'source_pro_ext_+1.csv')
    # Proportion('keywords_ext_-1.txt', 'keywords_pro_ext_-1.csv')
    # Proportion('keywords_ext_+1.txt', 'keywords_pro_ext_+1.csv')






