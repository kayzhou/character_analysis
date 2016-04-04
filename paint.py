# -*- coding: utf-8 -*-
__author__ = 'Kay'

import matplotlib.pylab as plt
import pandas as pd
import numpy as np


def cumulative_hist():
    n_bins = 5000
    data1 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_n1_pro.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_1_pro.txt', sep=' ', header=None)

    shopping1 = data1[2]; shopping2 = data2[2]
    for i in np.arange(3, 17):
        shopping1 += data1[i]
        shopping2 += data2[i]

    # print(shopping1)
    # print(shopping2)

    col1 = shopping1 / data1[1]
    col2 = shopping2 / data2[1]
    # col1 = col1[col1 > 0]
    # col2 = col2[col2 > 0]

    print(col2)
    plt.hist(col1, n_bins, normed=1, alpha=0.6, color='b', linewidth=1.5, histtype='step', cumulative=True)
    plt.hist(col2, n_bins, normed=1, alpha=0.6, color='r', linewidth=1.5, histtype='step', cumulative=True)
    plt.xlim(0, 0.15)
    plt.ylim(0, 1)
    # plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
    # plt.hist(data2[1], alpha=0.6, color='r')
    plt.show()


def ca_box_plot_shopping():
    data1 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_n1_pro.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_401_shopping_1_pro.txt', sep=' ', header=None)

    shopping1 = data1[2]; shopping2 = data2[2]
    for i in np.arange(3, 17):
        shopping1 += data1[i]
        shopping2 += data2[i]

    col1 = shopping1 / data1[1]
    col2 = shopping2 / data2[1]
    col1 = col1[col1 > 0][col1 < 0.2]
    col2 = col2[col2 > 0][col2 < 0.2]
    print(col1.describe())
    print(col2.describe())
    re = plt.boxplot([col1, col2], showmeans=True, showfliers=False)
    plt.show()


def ca_box_plot_features(index):
    data1 = pd.read_csv('data/split_class/large_IGNORE_401_features_n1_pro.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_401_features_1_pro.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]

    plt.boxplot([col1, col2], showmeans=True, showfliers=False)
    plt.show()


def ca_hist_features(index):
    data1 = pd.read_csv('data/split_class/large_IGNORE_401_features_n1_pro.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_401_features_1_pro.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]

    plt.subplot(1, 2, 1)
    plt.suptitle("introversion")
    plt.hist(col1, alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.suptitle("extraversion")
    plt.hist(col2, alpha=0.6)
    plt.show()


if __name__ == '__main__':
    # ca_hist_features(1)
    ca_box_plot_features(5)