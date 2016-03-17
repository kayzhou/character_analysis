# -*- coding: utf-8 -*-
__author__ = 'Kay'

import codecs

for line in codecs.open('NLP_data/stop_word.txt', 'r').readlines():
    print(line.strip())