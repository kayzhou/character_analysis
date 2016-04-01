# -*- coding: utf-8 -*-
__author__ = 'Kay'

import matplotlib.pylab as plt
import pandas as pd

data = pd.read_csv('data/large_401_shopping.txt', sep=' ', header=None)
plt.hist(data[1], alpha=0.6, color='b')
plt.hist(data[2], alpha=0.6, color='r')
plt.show()