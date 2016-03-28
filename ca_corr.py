# -*- coding: utf-8 -*-
__author__ = 'Kay'

import pandas as pd

raw_data = pd.read_csv('data/for_analysis/328_features_regress_0.txt', delimiter=' ', header=None)
raw_data.corr()[0].to_csv('data/for_analysis/corr_regress_0.csv')

raw_data = pd.read_csv('data/for_analysis/328_features_regress_1.txt', delimiter=' ', header=None)
raw_data.corr()[0].to_csv('data/for_analysis/corr_regress_1.csv')

raw_data = pd.read_csv('data/for_analysis/328_features_regress_2.txt', delimiter=' ', header=None)
raw_data.corr()[0].to_csv('data/for_analysis/corr_regress_2.csv')

raw_data = pd.read_csv('data/for_analysis/328_features_regress_3.txt', delimiter=' ', header=None)
raw_data.corr()[0].to_csv('data/for_analysis/corr_regress_3.csv')

raw_data = pd.read_csv('data/for_analysis/328_features_regress_4.txt', delimiter=' ', header=None)
raw_data.corr()[0].to_csv('data/for_analysis/corr_regress_4.csv')
