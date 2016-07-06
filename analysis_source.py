# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
import pandas as pd

def get_brand_count(in_name):
    brand = {
        "苹果": ["iphone", "ipad", "ios"],
        "三星": ["三星", "samsung", "galaxy"],
        "小米": ["小米", "红米"],
        "华为": ["华为", "荣耀"],
        "锤子": ["坚果", "smartisan"],
        "乐视": ["乐视"],
        "联想": ["联想"],
        "魅族": ["魅族", "魅蓝"],
        "诺基亚": ["诺基亚", "lumia"],
        "索尼": ["索尼"],
        "htc": ["htc"],
        "oppo": ["oppo"],
        "vivo": ["vivo"],
        "nubia": ["nubia"],
        "lg": ["lg"],
        "android": ["android"]
    }

    brand_count = {
        "苹果": 0,
        "三星": 0,
        "小米": 0,
        "华为": 0,
        "锤子": 0,
        "乐视": 0,
        "联想": 0,
        "魅族": 0,
        "诺基亚": 0,
        "索尼": 0,
        "htc": 0,
        "oppo": 0,
        "vivo": 0,
        "nubia": 0,
        "lg": 0,
        "android": 0
    }

    data = pd.read_csv(in_name)

    # 遍历所有来源
    for i in np.arange(data.shape[0]):
        bingo = 0
        item = data.ix[i, :]
        # 遍历所有品牌
        for bra in brand.items():
            # 遍历所有型号
            for typ in bra[1]:
                if typ in item.source.lower():
                    brand_count[bra[0]] += item['count']
                    bingo = 1
                    # if bra[0] == "苹果":
                    #     print(item['source'])
                    break

        if not bingo and item['count'] > 300:
            print(item['source'])
    print('----------')

    # 排序
    brand_count_sorted = sorted(brand_count.items(), key=lambda d:d[1], reverse = True)

    # 计算总量
    s = 0
    for c in brand_count.items():
        s += c[1]

    # 打印输出
    for bcs in brand_count_sorted:
        print(bcs[0] + ',' + str(bcs[1]) + ',' + "%.4f" % (bcs[1] / s))

    # out.to_csv(out_name, encoding='utf8')

def get_brand_count(in_name):
    brand = {
        "苹果": ["iphone", "ipad", "ios"],
        "三星": ["三星", "samsung", "galaxy"],
        "小米": ["小米", "红米"],
        "华为": ["华为", "荣耀"],
        "锤子": ["坚果", "smartisan"],
        "乐视": ["乐视"],
        "联想": ["联想"],
        "魅族": ["魅族", "魅蓝"],
        "诺基亚": ["诺基亚", "lumia"],
        "索尼": ["索尼"],
        "htc": ["htc"],
        "oppo": ["oppo"],
        "vivo": ["vivo"],
        "nubia": ["nubia"],
        "lg": ["lg"],
        "android": ["android"]
    }

    brand_count = {
        "苹果": 0,
        "三星": 0,
        "小米": 0,
        "华为": 0,
        "锤子": 0,
        "乐视": 0,
        "联想": 0,
        "魅族": 0,
        "诺基亚": 0,
        "索尼": 0,
        "htc": 0,
        "oppo": 0,
        "vivo": 0,
        "nubia": 0,
        "lg": 0,
        "android": 0
    }

    data = pd.read_csv(in_name)

    # 遍历所有来源
    for i in np.arange(data.shape[0]):
        bingo = 0
        item = data.ix[i, :]
        # 遍历所有品牌
        for bra in brand.items():
            # 遍历所有型号
            for typ in bra[1]:
                if typ in item.source.lower():
                    brand_count[bra[0]] += item['count']
                    bingo = 1
                    # if bra[0] == "苹果":
                    #     print(item['source'])
                    break

        if not bingo and item['count'] > 300:
            print(item['source'])
    print('----------')

    # 排序
    brand_count_sorted = sorted(brand_count.items(), key=lambda d:d[1], reverse = True)

    # 计算总量
    s = 0
    for c in brand_count.items():
        s += c[1]

    # 打印输出
    for bcs in brand_count_sorted:
        print(bcs[0] + ',' + str(bcs[1]) + ',' + "%.4f" % (bcs[1] / s))

    # out.to_csv(out_name, encoding='utf8')

def get_source_count(in_name):
    source = {
        # "浏览器": ["浏览器", "weibo.com"],
        "自拍": ["美图", "美拍"],
        "新闻": ["新闻", "今日头条"],
        "视频": ["秒拍", "哔哩哔哩", "优酷土豆", "爱奇艺"],
        "知乎": ["知乎"],
        "豆瓣": ["豆瓣"],
        "音乐": ["音乐", "唱吧"],
        "购物": ["淘宝", "亚马逊"]
    }

    source_pro = {
        # "浏览器": 0,
        "自拍": 0,
        "新闻": 0,
        "视频": 0,
        "知乎": 0,
        "豆瓣": 0,
        "音乐": 0,
        "购物": 0
    }

    data = pd.read_csv(in_name)


    # 遍历所有来源
    for i in np.arange(data.shape[0]):
        item = data.ix[i, :]
        # 遍历所有来源类别
        for sou in source.items():
            # 遍历所有具体来源
            for typ in sou[1]:
                if typ in item.source.lower():
                    source_pro[sou[0]] += item['count']
                    break

    print('----------')


    # 计算总量
    s = 0
    for c in source_pro.items():
        s += c[1]

    # 打印输出
    for sp in source_pro.items():
        # print(sp[0] + ',' + "%.4f" % (sp[1] * 100) +"%")
        print(sp[0] + ',' + "%.2f" % (sp[1] / s * 100) +"%")

    # out.to_csv(out_name, encoding='utf8')

if __name__ == '__main__':
    # get_brand_count("data/features/source_pro_ext_+1.csv")
    # get_brand_count("data/features/source_pro_ext_-1.csv")

    get_source_count("data/features/source_pro_ext_+1.csv")
    get_source_count("data/features/source_pro_ext_-1.csv")