#coding:utf-8
import sys
import separate
import os
import math

emotion_count=5
feature_dict={}
#pre_list=[math.log(1.0/emotion_count)]*emotion_count
pre_list = [0.2,0.2,0.2,0.2,0.2]
pre_list = [math.log(pro) for pro in pre_list]

def load_feature():
    global feature_dict
    for line in open('words_pre_pro.last'):
        line_arr=line.strip().split('\t')
        feature=line_arr[0]
        pre_pro_list=[float(pre_pro) for pre_pro in line_arr[1:]]
        feature_dict[feature]=pre_pro_list


# 输入为utf8格式的经分词处理后的字符串数组
def classify(seg):
    word_list=[word for word in seg if word and not word=='#']
    if len(word_list) <= 2:
        return -1
    emotion_list=list(pre_list)
    is_true=False
    for word in word_list:
        try:
            pre_pro_list=feature_dict[word]
            is_true=True
            for emotion in range(emotion_count):
                try:
                    emotion_list[emotion]+=math.log(pre_pro_list[emotion])
                except ValueError:
                    raise
        except KeyError:
            pass
    if not is_true:
        return -1
    max_emotion=0
    for emotion in range(emotion_count):
        if emotion_list[emotion]!=0 and emotion_list[emotion]>emotion_list[max_emotion]:
            max_emotion=emotion
    return max_emotion

load_feature()