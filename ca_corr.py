# -*- coding: utf-8 -*-
__author__ = 'Kay'

import pandas as pd
import matplotlib.pyplot as plt


def ca_box_plot_features(data_list):
    plt.xlabel('character')
    plt.ylabel('correlation')
    plt.boxplot(data_list)
    plt.show()


raw_data1 = pd.read_csv('data/for_analysis/331_IGNORE_0.txt', delimiter=' ', header=None)
raw_data1.corr()[0][1:].to_csv('data/for_analysis/corr_regress_0.csv')

raw_data2 = pd.read_csv('data/for_analysis/331_IGNORE_1.txt', delimiter=' ', header=None)
raw_data2.corr()[0][1:].to_csv('data/for_analysis/corr_regress_1.csv')

raw_data3 = pd.read_csv('data/for_analysis/331_IGNORE_2.txt', delimiter=' ', header=None)
raw_data3.corr()[0][1:].to_csv('data/for_analysis/corr_regress_2.csv')

raw_data4 = pd.read_csv('data/for_analysis/331_IGNORE_3.txt', delimiter=' ', header=None)
raw_data4.corr()[0][1:].to_csv('data/for_analysis/corr_regress_3.csv')

raw_data5 = pd.read_csv('data/for_analysis/331_IGNORE_4.txt', delimiter=' ', header=None)
raw_data5.corr()[0][1:].to_csv('data/for_analysis/corr_regress_4.csv')


ca_box_plot_features([raw_data1.corr()[0][1:], raw_data2.corr()[0][1:], raw_data3.corr()[0][1:], raw_data4.corr()[0][1:], raw_data5.corr()[0][1:]])