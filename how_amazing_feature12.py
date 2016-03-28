# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np

for line in open("/Users/Kay/Project/EXP/character_analysis/data/SVM/315_features_3_sides.txt"):
    line = line.strip()
    values = line.split(" ")
    # for v in values:
    #     if v.startswith("12:"):
    #         values.remove(v)
    # print(" ".join(values))
    # print(values[0], "1:" + values[12][3:])
    if float(values[12][3:]) <= 20:
        print(values[0], values[12][3:])