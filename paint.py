# -*- coding: utf-8 -*-
__author__ = 'Kay'

import matplotlib.pylab as plt
import pandas as pd
import numpy as np

n_bins = 5000
data1 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_n1.txt', sep=' ', header=None)
data2 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_1.txt', sep=' ', header=None)

shopping1 = data1[2]; shopping2 = data2[2]
for i in np.arange(3, 17):
    shopping1 += data1[i]
    shopping2 += data2[i]

# print(shopping1)
# print(shopping2)

col1 = shopping1 / data1[1]
col2 = shopping2 / data2[1]
col1 = col1[col1 > 0]
col2 = col2[col2 > 0]

print(col2)
plt.hist(col1, n_bins, normed=1, alpha=0.6, color='b', linewidth=1.5, histtype='step', cumulative=True)
plt.hist(col2, n_bins, normed=1, alpha=0.6, color='r', linewidth=1.5, histtype='step', cumulative=True)
plt.xlim(0, 0.15)
plt.ylim(0, 1)
# plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
# plt.hist(data2[1], alpha=0.6, color='r')
plt.show()