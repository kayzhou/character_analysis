# -*- coding: utf-8 -*-
__author__ = 'Kay'

import pandas as pd

data = pd.read_csv("data/pro_401.txt", sep=" ", header=None)

new_data = data[data[1] > 0.35][0]
new_data.to_csv('data/asd.txt')

uid = set()

cnt = 0
for line in open('data/asd.txt'):
    cnt += 1
    uid.add(line.strip().split(',')[1])
    print(line.strip().split(',')[1] + " -1.0")

for line in open('/Users/Kay/Project/EXP/character_analysis/data/tags/large_tag_311_1.txt'):
    if line.strip().split(' ')[1] == "1.0":
        cnt += 1
        uid.add(line.strip().split(' ')[0])
        print(line.strip())

# print(len(uid) == cnt)
# print(len(uid), cnt)
