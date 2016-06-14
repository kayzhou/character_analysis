# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd
import math
import os
from geopy.geocoders import Nominatim
import json


def read_POI_file(in_name = 'POI_data/poi_list_2.txt'):
    # X -> 经度, Y -> 纬度
    data = pd.read_csv(in_name, header = None, sep='\t', names=['city', 'name', 'site', 'category', 'dis', 'Y', 'X'])
    return data


def read_POI_file_csv(in_dir = 'POI_data'):
    # X -> 经度, Y -> 纬度
    lines = 0
    data = []
    # data = pd.DataFrame({})
    for in_name in os.listdir(in_dir):
        print(in_name)
        if in_name.endswith(".csv"): # 是有效csv文件
            # raw = pd.read_csv(os.path.join(in_dir,in_name), encoding='gbk')
            raw = pd.read_csv(os.path.join(in_dir,in_name), encoding='utf8', usecols=['CLA', 'X', 'Y'])
            data.append({"raw": raw, "min_X": min(raw.X), "max_X": max(raw.X), "min_Y": min(raw.Y), "max_Y": max(raw.Y)})
            # raw = raw.reindex(np.arange(lines, lines+raw.shape[0]))
            # lines += raw.shape[0]
            # data = pd.concat([data, raw], axis=0)
    return data


def get_geo_raw(x_y):
    try:
        r = json.dumps(Nominatim().reverse(x_y, timeout = 10).raw)
        return r
    except:
        return None


def get_distance(lat_a, lng_a, lat_b, lng_b):
    '''
    获取两地距离
    :param lat_a: a纬度
    :param lng_a: a经度
    :param lat_b: b纬度
    :param lng_b: b经度
    :return: 两地距离 (米)
    '''
    pk = 180 / 3.14169
    a1 = lat_a / pk
    a2 = lng_a / pk
    b1 = lat_b / pk
    b2 = lng_b / pk
    t1 = math.cos(a1) * math.cos(a2) * math.cos(b1) * math.cos(b2)
    t2 = math.cos(a1) * math.sin(a2) * math.cos(b1) * math.sin(b2)
    t3 = math.sin(a1) * math.sin(b1)
    tt = math.acos(t1 + t2 + t3)
    return 6366000 * tt


def get_site_category(x, y, POI_list):

    # 北京市
    # if y < 39 or y > 41 or x < 115 or x > 118:
    #     return None
    # else:
    #    return "bingo"
    
    print(x, y)

    for meta in POI_list:
        if meta['min_X'] <= x <= meta['max_X'] and meta['min_Y'] <= y <= meta['max_Y']:
            POI = meta['raw']
            for i in np.arange(POI.shape[0]):
                # print (POI.X[i] - x, POI.Y[i] - y)
                # print(POI.ix[i, :])
                if abs(POI.X[i] - x) <= 0.001 and abs(POI.Y[i] - y) <= 0.001:
                    # return "bingo"
                    # return POI.city[i] + ":" + POI.category[i]
                    return POI.CLA[i]


def geo(in_name, out_name):
    out_file = open(out_name, 'w', encoding='utf8')
    for line in open(in_name, encoding='utf8').readlines():
        location = []
        uid = line[:10]
        print(uid)
        if not line[10: ].strip():
            continue
        list_y_x = line[10: ].strip().split(' ')
        print(list_y_x)
        for y_x in list_y_x:
            print(y_x)
            r = get_geo_raw(y_x)
            if r:
                location.append(r)
        out_file.write(uid + '\n' + '\n'.join(location) + '\n')


def locate(in_name, out_name):
    POI = read_POI_file_csv()
    out_file = open(out_name, 'w', encoding='utf8')
    remember = {}
    ignore = set()
    bingo = 0
    for line in open(in_name, encoding='utf8').readlines():
        # cnt += 1
        # if cnt < 1141: continue
        location = []
        uid = line[:11]
        print(uid)
        list_y_x = line[11: ].strip().split(' ')
        for y_x in list_y_x:
            
            if y_x in remember:
                location.append(remember[y_x])
                bingo += 1
                continue
            if y_x == "" or y_x in ignore:
                continue

            # print(y_x)
            y, x = y_x.split(',')
            re = get_site_category(float(x), float(y), POI)
            print('bingo =', bingo)

            if re:
                location.append(re)
                remember[y_x] = re
                bingo += 1
            else:
                ignore.add(y_x)

        out_file.write(uid + ' '.join(location) + '\n')


def analysis_POI(in_name):
    # 每行是一个用户
    stat = {}
    _sum = 0
    for line in open(in_name, encoding='utf8').readlines():
        uid = line[:11]
        # print(uid)
        if line[12: ] != '':
            list_geo = line[11: ].strip().split(' ')
            for geo in list_geo:
                # print(geo)
                # print(type(geo))
                try:
                    start = geo.index(',') + 1
                    end = geo.index(']')
                    geo_type = geo[start: end]
                except ValueError:
                    # print('格式出错!', geo)
                    continue
                _sum += 1
                if geo_type in stat:
                    stat[geo_type] += 1
                else:
                    stat[geo_type] = 1

    stat = sorted(stat.items(), key=lambda d:d[1], reverse = True)
    for it in stat:
        if it[1] > 40:
            print(it[0], it[1], '%.2f%%' % (it[1] / _sum * 100))
    print('---')
    # print(stat)


def analysis_geo(in_name):
    # 每行是一个用户
    _sum = 0
    user_city_count = {}
    for line in open(in_name, encoding='utf8').readlines():
        # print(len(line))
        if len(line) == 11:
            uid = line.strip()
            user_city_count[uid] = {}
            continue

        # print(type(json.loads(line.strip())))
        raw = json.loads(line.strip())
        # print(raw)
        # print(type(raw))
        try:
            if 'address' in raw:
                if 'city' in raw['address']:
                    loc = raw['address']['city']
                elif 'county' in raw['address']:
                    loc = raw['address']['county']

                if loc in user_city_count:
                    user_city_count[uid][loc] += 1
                else:
                    user_city_count[uid][loc] = 1

        except:
            print(raw)

    stat_count = {}
    for user_city in user_city_count.items():
        uid = user_city[0]
        city = user_city[1]
        le = len(city.items())
        # 分五类进行讨论
        if le == 1:
            group = '1'
        elif le == 2:
            group = '2'
        elif 2 < le <=5:
            group = '3-5'
        elif 5 < le <= 10:
            group = '6-10'
        elif 10 < le <= 20:
            group = '10-20'
        else:
            group = '>20'
        _sum += 1

        if group in stat_count:
            stat_count[group] += 1
        else:
            stat_count[group] = 1


    stat = sorted(stat_count.items(), key=lambda d:d[1], reverse = True)
    for it in stat:
        print(it[0], it[1], '%.2f%%' % (it[1] / _sum * 100))
    print('---')

    #         list_geo = line[11: ].strip().split(' ')
    #         for geo in list_geo:
    #             # print(geo)
    #             # print(type(geo))
    #             try:
    #                 start = geo.index(',') + 1
    #                 end = geo.index(']')
    #                 geo_type = geo[start: end]
    #             except ValueError:
    #                 # print('格式出错!', geo)
    #                 continue
    #             _sum += 1
    #             if geo_type in stat:
    #                 stat[geo_type] += 1
    #             else:
    #                 stat[geo_type] = 1
    #
    # stat = sorted(stat.items(), key=lambda d:d[1], reverse = True)
    # for it in stat:
    #     if it[1] > 40:
    #         print(it[0], it[1], '%.2f%%' % (it[1] / _sum * 100))
    # print('---')
    # print(stat)



if __name__ == '__main__':
    # read_POI_file_csv()
    # locate('data/large_517_geo_-1.txt', 'large_531_POI_-1_100M.txt')
    # locate('data/large_517_geo_+1.txt', 'large_531_POI_+1_100M.txt')
    # print(read_POI_file_new())
    # geo('data/large_517_geo_-1.txt', 'large_527_geo_-1.txt')
    # geo('data/large_517_geo_+1.txt', 'large_527_geo_+1.txt')

    # analysis_POI('data/features/large_531_POI_+1_100M.txt')
    # analysis_POI('data/features/large_531_POI_-1_100M.txt')


    analysis_geo('/Users/Kay/Project/EXP/character_analysis/data/features/large_527_geo_+1.txt')
    analysis_geo('/Users/Kay/Project/EXP/character_analysis/data/features/large_527_geo_-1.txt')