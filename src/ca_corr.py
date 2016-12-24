# -*- coding: utf-8 -*-
__author__ = 'Kay'

import pandas as pd
import matplotlib.pyplot as plt


def ca_box_plot_features(data_list):
    plt.xlabel('character')
    plt.ylabel('correlation')
    plt.boxplot(data_list)
    plt.show()


# raw_data1 = pd.read_csv('data/for_analysis/404_IGNORE_NOR_0.txt', delimiter=' ', header=None)
# raw_data1.corr()[0][1:].to_csv('data/for_analysis/404_corr_0.csv')

raw_data2 = pd.read_csv('data/for_analysis/404_IGNORE_NOR_1.txt', delimiter=' ', header=None)
raw_data2.corr('spearman')[0][1:].to_csv('data/for_analysis/404_corr_1_spearman.csv')

# raw_data3 = pd.read_csv('data/for_analysis/404_IGNORE_NOR_2.txt', delimiter=' ', header=None)
# raw_data3.corr()[0][1:].to_csv('data/for_analysis/404_corr_2.csv')

# raw_data4 = pd.read_csv('data/for_analysis/404_IGNORE_NOR_3.txt', delimiter=' ', header=None)
# raw_data4.corr()[0][1:].to_csv('data/for_analysis/404_corr_3.csv')

# raw_data5 = pd.read_csv('data/for_analysis/404_IGNORE_NOR_4.txt', delimiter=' ', header=None)
# raw_data5.corr()[0][1:].to_csv('data/for_analysis/404_corr_4.csv')


# ca_box_plot_features([raw_data1.corr()[0][1:], raw_data2.corr()[0][1:], raw_data3.corr()[0][1:], raw_data4.corr()[0][1:], raw_data5.corr()[0][1:]])